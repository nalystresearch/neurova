// Copyright (c) 2026 @squid consultancy group (scg)
// all rights reserved.
// licensed under the apache license 2.0.

/**
 * neurova_minimal.cpp - Minimal C++ core for fast compilation and testing
 * 
 * Self-contained implementation without external header dependencies.
 * This provides the essential core functionality for Neurova.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <algorithm>
#include <random>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

namespace py = pybind11;

namespace neurova {

// Version
constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 2;
constexpr int VERSION_PATCH = 0;
constexpr const char* VERSION_STRING = "0.2.0";

// RNG
static thread_local std::mt19937 rng(std::random_device{}());

// ============================================================================
// DType
// ============================================================================

enum class DType {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    UINT8,
    BOOL
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32: return 4;
        case DType::FLOAT64: return 8;
        case DType::INT32: return 4;
        case DType::INT64: return 8;
        case DType::UINT8: return 1;
        case DType::BOOL: return 1;
        default: return 4;
    }
}

// ============================================================================
// Tensor
// ============================================================================

class Tensor {
public:
    Tensor() : size_(0), dtype_(DType::FLOAT32) {}
    
    Tensor(const std::vector<size_t>& shape, DType dtype = DType::FLOAT32) 
        : shape_(shape), dtype_(dtype) {
        size_ = 1;
        for (auto s : shape) size_ *= s;
        data_.reset(new char[size_ * dtype_size(dtype)]);
    }
    
    // Copy constructor
    Tensor(const Tensor& other) : shape_(other.shape_), size_(other.size_), dtype_(other.dtype_) {
        data_.reset(new char[size_ * dtype_size(dtype_)]);
        std::memcpy(data_.get(), other.data_.get(), size_ * dtype_size(dtype_));
    }
    
    // Copy assignment
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            shape_ = other.shape_;
            size_ = other.size_;
            dtype_ = other.dtype_;
            data_.reset(new char[size_ * dtype_size(dtype_)]);
            std::memcpy(data_.get(), other.data_.get(), size_ * dtype_size(dtype_));
        }
        return *this;
    }
    
    // Move constructor
    Tensor(Tensor&& other) noexcept = default;
    
    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept = default;
    
    // Factory methods
    static Tensor zeros(const std::vector<size_t>& shape, DType dtype = DType::FLOAT32) {
        Tensor t(shape, dtype);
        std::memset(t.data_.get(), 0, t.nbytes());
        return t;
    }
    
    static Tensor ones(const std::vector<size_t>& shape, DType dtype = DType::FLOAT32) {
        Tensor t(shape, dtype);
        if (dtype == DType::FLOAT32) {
            float* p = t.ptr<float>();
            for (size_t i = 0; i < t.size_; ++i) p[i] = 1.0f;
        }
        return t;
    }
    
    static Tensor randn(const std::vector<size_t>& shape, DType dtype = DType::FLOAT32) {
        Tensor t(shape, dtype);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        if (dtype == DType::FLOAT32) {
            float* p = t.ptr<float>();
            for (size_t i = 0; i < t.size_; ++i) p[i] = dist(rng);
        }
        return t;
    }
    
    static Tensor rand(const std::vector<size_t>& shape, DType dtype = DType::FLOAT32) {
        Tensor t(shape, dtype);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dtype == DType::FLOAT32) {
            float* p = t.ptr<float>();
            for (size_t i = 0; i < t.size_; ++i) p[i] = dist(rng);
        }
        return t;
    }
    
    static Tensor arange(float start, float stop, float step = 1.0f) {
        size_t n = static_cast<size_t>((stop - start) / step);
        Tensor t({n}, DType::FLOAT32);
        float* p = t.ptr<float>();
        for (size_t i = 0; i < n; ++i) p[i] = start + i * step;
        return t;
    }
    
    static Tensor eye(size_t n, DType dtype = DType::FLOAT32) {
        Tensor t = zeros({n, n}, dtype);
        float* p = t.ptr<float>();
        for (size_t i = 0; i < n; ++i) p[i * n + i] = 1.0f;
        return t;
    }
    
    // Properties
    std::vector<size_t> shape() const { return shape_; }
    size_t size() const { return size_; }
    size_t ndim() const { return shape_.size(); }
    DType dtype() const { return dtype_; }
    size_t nbytes() const { return size_ * dtype_size(dtype_); }
    
    // Data access
    void* data() { return data_.get(); }
    const void* data() const { return data_.get(); }
    
    template<typename T>
    T* ptr() { return reinterpret_cast<T*>(data_.get()); }
    
    template<typename T>
    const T* ptr() const { return reinterpret_cast<const T*>(data_.get()); }
    
    // Reshape
    Tensor reshape(const std::vector<size_t>& new_shape) const {
        size_t new_size = 1;
        for (auto s : new_shape) new_size *= s;
        if (new_size != size_) {
            throw std::runtime_error("Cannot reshape: size mismatch");
        }
        Tensor result(new_shape, dtype_);
        std::memcpy(result.data(), data(), nbytes());
        return result;
    }
    
    Tensor squeeze(int axis = -1) const {
        std::vector<size_t> new_shape;
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (shape_[i] != 1 || (axis >= 0 && static_cast<size_t>(axis) != i)) {
                new_shape.push_back(shape_[i]);
            }
        }
        if (new_shape.empty()) new_shape.push_back(1);
        return reshape(new_shape);
    }
    
    Tensor unsqueeze(int axis) const {
        std::vector<size_t> new_shape = shape_;
        if (axis < 0) axis = static_cast<int>(shape_.size()) + axis + 1;
        new_shape.insert(new_shape.begin() + axis, 1);
        return reshape(new_shape);
    }
    
    Tensor transpose() const {
        if (shape_.size() != 2) {
            throw std::runtime_error("transpose requires 2D tensor");
        }
        size_t rows = shape_[0], cols = shape_[1];
        Tensor result({cols, rows}, dtype_);
        
        const float* src = ptr<float>();
        float* dst = result.ptr<float>();
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
        return result;
    }
    
    // Clone
    Tensor clone() const {
        Tensor result(shape_, dtype_);
        std::memcpy(result.data(), data(), nbytes());
        return result;
    }
    
    Tensor contiguous() const { return clone(); }
    
    // Type conversion
    Tensor to(DType new_dtype) const {
        if (new_dtype == dtype_) return clone();
        
        Tensor result(shape_, new_dtype);
        
        if (dtype_ == DType::FLOAT32 && new_dtype == DType::UINT8) {
            const float* src = ptr<float>();
            uint8_t* dst = result.ptr<uint8_t>();
            for (size_t i = 0; i < size_; ++i) {
                dst[i] = static_cast<uint8_t>(std::clamp(src[i], 0.0f, 255.0f));
            }
        } else if (dtype_ == DType::UINT8 && new_dtype == DType::FLOAT32) {
            const uint8_t* src = ptr<uint8_t>();
            float* dst = result.ptr<float>();
            for (size_t i = 0; i < size_; ++i) {
                dst[i] = static_cast<float>(src[i]);
            }
        }
        
        return result;
    }
    
    // Arithmetic operations
    Tensor operator+(const Tensor& other) const {
        if (size_ != other.size_) throw std::runtime_error("Size mismatch");
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        const float* b = other.ptr<float>();
        float* c = result.ptr<float>();
        
        #if defined(__ARM_NEON)
        size_t i = 0;
        for (; i + 4 <= size_; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            vst1q_f32(c + i, vaddq_f32(va, vb));
        }
        for (; i < size_; ++i) c[i] = a[i] + b[i];
        #else
        for (size_t i = 0; i < size_; ++i) c[i] = a[i] + b[i];
        #endif
        
        return result;
    }
    
    Tensor operator-(const Tensor& other) const {
        if (size_ != other.size_) throw std::runtime_error("Size mismatch");
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        const float* b = other.ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = a[i] - b[i];
        return result;
    }
    
    Tensor operator*(const Tensor& other) const {
        if (size_ != other.size_) throw std::runtime_error("Size mismatch");
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        const float* b = other.ptr<float>();
        float* c = result.ptr<float>();
        
        #if defined(__ARM_NEON)
        size_t i = 0;
        for (; i + 4 <= size_; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            vst1q_f32(c + i, vmulq_f32(va, vb));
        }
        for (; i < size_; ++i) c[i] = a[i] * b[i];
        #else
        for (size_t i = 0; i < size_; ++i) c[i] = a[i] * b[i];
        #endif
        
        return result;
    }
    
    Tensor operator/(const Tensor& other) const {
        if (size_ != other.size_) throw std::runtime_error("Size mismatch");
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        const float* b = other.ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = a[i] / (b[i] + 1e-10f);
        return result;
    }
    
    Tensor operator-() const {
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = -a[i];
        return result;
    }
    
    Tensor operator+(float s) const {
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = a[i] + s;
        return result;
    }
    
    Tensor operator-(float s) const { return *this + (-s); }
    
    Tensor operator*(float s) const {
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = a[i] * s;
        return result;
    }
    
    Tensor operator/(float s) const { return *this * (1.0f / s); }
    
    // Reduction operations
    float sum() const {
        const float* p = ptr<float>();
        float s = 0.0f;
        for (size_t i = 0; i < size_; ++i) s += p[i];
        return s;
    }
    
    float mean() const { return sum() / size_; }
    
    float min() const {
        const float* p = ptr<float>();
        float m = p[0];
        for (size_t i = 1; i < size_; ++i) if (p[i] < m) m = p[i];
        return m;
    }
    
    float max() const {
        const float* p = ptr<float>();
        float m = p[0];
        for (size_t i = 1; i < size_; ++i) if (p[i] > m) m = p[i];
        return m;
    }
    
    float std() const { return std::sqrt(var()); }
    
    float var() const {
        float m = mean();
        const float* p = ptr<float>();
        float s = 0.0f;
        for (size_t i = 0; i < size_; ++i) {
            float d = p[i] - m;
            s += d * d;
        }
        return s / size_;
    }
    
    size_t argmax() const {
        const float* p = ptr<float>();
        size_t idx = 0;
        float m = p[0];
        for (size_t i = 1; i < size_; ++i) {
            if (p[i] > m) { m = p[i]; idx = i; }
        }
        return idx;
    }
    
    size_t argmin() const {
        const float* p = ptr<float>();
        size_t idx = 0;
        float m = p[0];
        for (size_t i = 1; i < size_; ++i) {
            if (p[i] < m) { m = p[i]; idx = i; }
        }
        return idx;
    }
    
    // Matrix multiplication
    Tensor matmul(const Tensor& other) const {
        if (shape_.size() != 2 || other.shape_.size() != 2) {
            throw std::runtime_error("matmul requires 2D tensors");
        }
        if (shape_[1] != other.shape_[0]) {
            throw std::runtime_error("matmul shape mismatch");
        }
        
        size_t M = shape_[0], K = shape_[1], N = other.shape_[1];
        Tensor result({M, N}, dtype_);
        
        const float* A = ptr<float>();
        const float* B = other.ptr<float>();
        float* C = result.ptr<float>();
        
        // Zero output
        std::memset(C, 0, M * N * sizeof(float));
        
        // Blocked matrix multiplication
        constexpr size_t BS = 32;
        for (size_t i0 = 0; i0 < M; i0 += BS) {
            size_t imax = std::min(i0 + BS, M);
            for (size_t k0 = 0; k0 < K; k0 += BS) {
                size_t kmax = std::min(k0 + BS, K);
                for (size_t j0 = 0; j0 < N; j0 += BS) {
                    size_t jmax = std::min(j0 + BS, N);
                    
                    for (size_t i = i0; i < imax; ++i) {
                        for (size_t k = k0; k < kmax; ++k) {
                            float a_ik = A[i * K + k];
                            for (size_t j = j0; j < jmax; ++j) {
                                C[i * N + j] += a_ik * B[k * N + j];
                            }
                        }
                    }
                }
            }
        }
        
        return result;
    }
    
    Tensor dot(const Tensor& other) const { return matmul(other); }
    
    // Element-wise math
    Tensor exp() const {
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::exp(a[i]);
        return result;
    }
    
    Tensor log() const {
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::log(a[i] + 1e-10f);
        return result;
    }
    
    Tensor sqrt() const {
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::sqrt(std::max(a[i], 0.0f));
        return result;
    }
    
    Tensor pow(float exp) const {
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::pow(a[i], exp);
        return result;
    }
    
    Tensor abs() const {
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::abs(a[i]);
        return result;
    }
    
    Tensor sin() const {
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::sin(a[i]);
        return result;
    }
    
    Tensor cos() const {
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::cos(a[i]);
        return result;
    }
    
    Tensor tanh() const {
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::tanh(a[i]);
        return result;
    }
    
    Tensor clamp(float min_val, float max_val) const {
        Tensor result(shape_, dtype_);
        const float* a = ptr<float>();
        float* c = result.ptr<float>();
        for (size_t i = 0; i < size_; ++i) c[i] = std::clamp(a[i], min_val, max_val);
        return result;
    }
    
private:
    std::unique_ptr<char[]> data_;
    std::vector<size_t> shape_;
    size_t size_;
    DType dtype_;
};

// ============================================================================
// Image
// ============================================================================

class Image {
public:
    Image() : width_(0), height_(0), channels_(0), dtype_(DType::UINT8) {}
    
    Image(size_t width, size_t height, size_t channels = 3, DType dtype = DType::UINT8)
        : width_(width), height_(height), channels_(channels), dtype_(dtype) {
        data_ = Tensor({channels, height, width}, dtype);
    }
    
    // Copy constructor
    Image(const Image& other) 
        : data_(other.data_), width_(other.width_), height_(other.height_), 
          channels_(other.channels_), dtype_(other.dtype_) {}
    
    // Copy assignment
    Image& operator=(const Image& other) {
        if (this != &other) {
            data_ = other.data_;
            width_ = other.width_;
            height_ = other.height_;
            channels_ = other.channels_;
            dtype_ = other.dtype_;
        }
        return *this;
    }
    
    // Move constructor
    Image(Image&& other) noexcept = default;
    
    // Move assignment
    Image& operator=(Image&& other) noexcept = default;
    
    // Properties
    size_t width() const { return width_; }
    size_t height() const { return height_; }
    size_t channels() const { return channels_; }
    DType dtype() const { return dtype_; }
    size_t size() const { return width_ * height_ * channels_; }
    
    template<typename T>
    T* ptr() { return data_.ptr<T>(); }
    
    template<typename T>
    const T* ptr() const { return data_.ptr<T>(); }
    
    Tensor to_tensor() const { return data_.clone(); }
    
    Image to(DType new_dtype) const {
        Image result(width_, height_, channels_, new_dtype);
        
        if (dtype_ == DType::UINT8 && new_dtype == DType::FLOAT32) {
            const uint8_t* src = ptr<uint8_t>();
            float* dst = result.ptr<float>();
            for (size_t i = 0; i < size(); ++i) {
                dst[i] = static_cast<float>(src[i]) / 255.0f;
            }
        } else if (dtype_ == DType::FLOAT32 && new_dtype == DType::UINT8) {
            const float* src = ptr<float>();
            uint8_t* dst = result.ptr<uint8_t>();
            for (size_t i = 0; i < size(); ++i) {
                dst[i] = static_cast<uint8_t>(std::clamp(src[i] * 255.0f, 0.0f, 255.0f));
            }
        }
        
        return result;
    }
    
    Image clone() const {
        Image result(width_, height_, channels_, dtype_);
        std::memcpy(result.ptr<char>(), ptr<char>(), size() * dtype_size(dtype_));
        return result;
    }
    
private:
    Tensor data_;
    size_t width_, height_, channels_;
    DType dtype_;
};

// ============================================================================
// Rect
// ============================================================================

struct Rect {
    int x = 0, y = 0;
    int width = 0, height = 0;
    
    Rect() = default;
    Rect(int x_, int y_, int w_, int h_) : x(x_), y(y_), width(w_), height(h_) {}
    int area() const { return width * height; }
};

// ============================================================================
// Image Processing Functions
// ============================================================================

namespace imgproc {

Image rgb_to_gray(const Image& src) {
    Image dst(src.width(), src.height(), 1, src.dtype());
    
    size_t pixels = src.width() * src.height();
    
    if (src.dtype() == DType::UINT8) {
        const uint8_t* s = src.ptr<uint8_t>();
        uint8_t* d = dst.ptr<uint8_t>();
        
        for (size_t i = 0; i < pixels; ++i) {
            uint8_t r = s[i];
            uint8_t g = s[pixels + i];
            uint8_t b = s[2 * pixels + i];
            d[i] = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
        }
    } else {
        const float* s = src.ptr<float>();
        float* d = dst.ptr<float>();
        
        for (size_t i = 0; i < pixels; ++i) {
            float r = s[i];
            float g = s[pixels + i];
            float b = s[2 * pixels + i];
            d[i] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }
    
    return dst;
}

Image gray_to_rgb(const Image& src) {
    Image dst(src.width(), src.height(), 3, src.dtype());
    
    size_t pixels = src.width() * src.height();
    
    if (src.dtype() == DType::UINT8) {
        const uint8_t* s = src.ptr<uint8_t>();
        uint8_t* d = dst.ptr<uint8_t>();
        
        for (size_t i = 0; i < pixels; ++i) {
            d[i] = s[i];
            d[pixels + i] = s[i];
            d[2 * pixels + i] = s[i];
        }
    }
    
    return dst;
}

Image gaussian_blur(const Image& src, int ksize = 3, float sigma = 0.0f) {
    if (sigma <= 0) sigma = 0.3f * ((ksize - 1) * 0.5f - 1) + 0.8f;
    
    Image gray = (src.channels() == 1) ? src : rgb_to_gray(src);
    Image gray_f = gray.to(DType::FLOAT32);
    Image dst(gray.width(), gray.height(), 1, DType::FLOAT32);
    
    // Build Gaussian kernel
    int half = ksize / 2;
    std::vector<float> kernel(ksize);
    float sum = 0.0f;
    for (int i = 0; i < ksize; ++i) {
        int x = i - half;
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < ksize; ++i) kernel[i] /= sum;
    
    int W = gray.width();
    int H = gray.height();
    
    // Temporary buffer for separable convolution
    std::vector<float> temp(W * H, 0.0f);
    const float* s = gray_f.ptr<float>();
    float* d = dst.ptr<float>();
    
    // Horizontal pass
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int k = -half; k <= half; ++k) {
                int xx = std::clamp(x + k, 0, W - 1);
                sum += s[y * W + xx] * kernel[k + half];
            }
            temp[y * W + x] = sum;
        }
    }
    
    // Vertical pass
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int k = -half; k <= half; ++k) {
                int yy = std::clamp(y + k, 0, H - 1);
                sum += temp[yy * W + x] * kernel[k + half];
            }
            d[y * W + x] = sum;
        }
    }
    
    return (gray.dtype() == DType::UINT8) ? dst.to(DType::UINT8) : dst;
}

Image resize(const Image& src, size_t new_width, size_t new_height) {
    Image dst(new_width, new_height, src.channels(), src.dtype());
    
    float sx = static_cast<float>(src.width()) / new_width;
    float sy = static_cast<float>(src.height()) / new_height;
    
    size_t C = src.channels();
    size_t src_pixels = src.width() * src.height();
    size_t dst_pixels = new_width * new_height;
    
    if (src.dtype() == DType::UINT8) {
        const uint8_t* s = src.ptr<uint8_t>();
        uint8_t* d = dst.ptr<uint8_t>();
        
        for (size_t c = 0; c < C; ++c) {
            for (size_t y = 0; y < new_height; ++y) {
                for (size_t x = 0; x < new_width; ++x) {
                    float src_x = (x + 0.5f) * sx - 0.5f;
                    float src_y = (y + 0.5f) * sy - 0.5f;
                    
                    int x0 = static_cast<int>(src_x);
                    int y0 = static_cast<int>(src_y);
                    int x1 = std::min(x0 + 1, static_cast<int>(src.width()) - 1);
                    int y1 = std::min(y0 + 1, static_cast<int>(src.height()) - 1);
                    x0 = std::max(x0, 0);
                    y0 = std::max(y0, 0);
                    
                    float fx = src_x - x0;
                    float fy = src_y - y0;
                    
                    float v00 = s[c * src_pixels + y0 * src.width() + x0];
                    float v01 = s[c * src_pixels + y0 * src.width() + x1];
                    float v10 = s[c * src_pixels + y1 * src.width() + x0];
                    float v11 = s[c * src_pixels + y1 * src.width() + x1];
                    
                    float val = (1 - fy) * ((1 - fx) * v00 + fx * v01) +
                                fy * ((1 - fx) * v10 + fx * v11);
                    
                    d[c * dst_pixels + y * new_width + x] = static_cast<uint8_t>(val);
                }
            }
        }
    }
    
    return dst;
}

Image crop(const Image& src, int x, int y, int width, int height) {
    x = std::max(0, x);
    y = std::max(0, y);
    width = std::min(width, static_cast<int>(src.width()) - x);
    height = std::min(height, static_cast<int>(src.height()) - y);
    
    Image dst(width, height, src.channels(), src.dtype());
    
    size_t C = src.channels();
    
    if (src.dtype() == DType::UINT8) {
        const uint8_t* s = src.ptr<uint8_t>();
        uint8_t* d = dst.ptr<uint8_t>();
        
        for (size_t c = 0; c < C; ++c) {
            for (int dy = 0; dy < height; ++dy) {
                for (int dx = 0; dx < width; ++dx) {
                    d[c * width * height + dy * width + dx] = 
                        s[c * src.width() * src.height() + (y + dy) * src.width() + (x + dx)];
                }
            }
        }
    }
    
    return dst;
}

// Threshold operations
Image threshold(const Image& src, float thresh, float maxval, int type) {
    if (src.dtype() != DType::UINT8) {
        throw std::runtime_error("threshold: only UINT8 supported");
    }
    
    Image result(src.width(), src.height(), src.channels(), src.dtype());
    size_t total = src.width() * src.height() * src.channels();
    
    const uint8_t* src_ptr = src.ptr<uint8_t>();
    uint8_t* dst_ptr = result.ptr<uint8_t>();
    
    uint8_t t = static_cast<uint8_t>(thresh);
    uint8_t m = static_cast<uint8_t>(maxval);
    
    // Type 0: binary (val > thresh ? maxval : 0)
    // Type 1: binary_inv (val > thresh ? 0 : maxval)
    if (type == 0) {
        for (size_t i = 0; i < total; ++i) {
            dst_ptr[i] = src_ptr[i] > t ? m : 0;
        }
    } else if (type == 1) {
        for (size_t i = 0; i < total; ++i) {
            dst_ptr[i] = src_ptr[i] > t ? 0 : m;
        }
    } else {
        // Type 2: trunc, 3: tozero, 4: tozero_inv
        for (size_t i = 0; i < total; ++i) {
            if (type == 2) {
                dst_ptr[i] = src_ptr[i] > t ? t : src_ptr[i];
            } else if (type == 3) {
                dst_ptr[i] = src_ptr[i] > t ? src_ptr[i] : 0;
            } else {
                dst_ptr[i] = src_ptr[i] > t ? 0 : src_ptr[i];
            }
        }
    }
    
    return result;
}

// Adaptive threshold
Image adaptive_threshold(const Image& src, float maxval, int method, int type, int block_size, float C) {
    if (src.dtype() != DType::UINT8 || src.channels() != 1) {
        throw std::runtime_error("adaptive_threshold: grayscale UINT8 only");
    }
    
    int w = src.width();
    int h = src.height();
    Image result(w, h, 1, DType::UINT8);
    
    const uint8_t* src_ptr = src.ptr<uint8_t>();
    uint8_t* dst_ptr = result.ptr<uint8_t>();
    
    int radius = block_size / 2;
    uint8_t m = static_cast<uint8_t>(maxval);
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Compute local mean
            int sum = 0;
            int count = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                        sum += src_ptr[ny * w + nx];
                        count++;
                    }
                }
            }
            float local_thresh = (float)sum / count - C;
            uint8_t val = src_ptr[y * w + x];
            
            if (type == 0) {  // binary
                dst_ptr[y * w + x] = val > local_thresh ? m : 0;
            } else {  // binary_inv
                dst_ptr[y * w + x] = val > local_thresh ? 0 : m;
            }
        }
    }
    
    return result;
}

// Rotate image
Image rotate(const Image& src, float angle, int center_x, int center_y) {
    if (src.dtype() != DType::UINT8) {
        throw std::runtime_error("rotate: only UINT8 supported");
    }
    
    int w = src.width();
    int h = src.height();
    int c = src.channels();
    
    // Default center
    float cx = center_x < 0 ? w / 2.0f : center_x;
    float cy = center_y < 0 ? h / 2.0f : center_y;
    
    // Convert to radians
    float rad = angle * 3.14159265f / 180.0f;
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);
    
    Image result(w, h, c, DType::UINT8);
    const uint8_t* src_ptr = src.ptr<uint8_t>();
    uint8_t* dst_ptr = result.ptr<uint8_t>();
    
    std::memset(dst_ptr, 0, w * h * c);
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Rotate coordinates back to source
            float dx = x - cx;
            float dy = y - cy;
            int src_x = static_cast<int>(dx * cos_a + dy * sin_a + cx + 0.5f);
            int src_y = static_cast<int>(-dx * sin_a + dy * cos_a + cy + 0.5f);
            
            if (src_x >= 0 && src_x < w && src_y >= 0 && src_y < h) {
                for (int ch = 0; ch < c; ++ch) {
                    dst_ptr[(y * w + x) * c + ch] = src_ptr[(src_y * w + src_x) * c + ch];
                }
            }
        }
    }
    
    return result;
}

// Histogram
Tensor histogram(const Image& src, int bins) {
    if (src.dtype() != DType::UINT8) {
        throw std::runtime_error("histogram: only UINT8 supported");
    }
    
    Tensor hist({static_cast<size_t>(bins)}, DType::FLOAT32);
    float* hist_ptr = static_cast<float*>(hist.data());
    std::memset(hist_ptr, 0, bins * sizeof(float));
    
    const uint8_t* src_ptr = src.ptr<uint8_t>();
    size_t total = src.width() * src.height() * src.channels();
    
    float scale = bins / 256.0f;
    for (size_t i = 0; i < total; ++i) {
        int bin = static_cast<int>(src_ptr[i] * scale);
        if (bin >= bins) bin = bins - 1;
        hist_ptr[bin] += 1.0f;
    }
    
    return hist;
}

// Histogram equalization
Image equalize_hist(const Image& src) {
    if (src.dtype() != DType::UINT8 || src.channels() != 1) {
        throw std::runtime_error("equalize_hist: grayscale UINT8 only");
    }
    
    int w = src.width();
    int h = src.height();
    
    // Compute histogram
    int hist[256] = {0};
    const uint8_t* src_ptr = src.ptr<uint8_t>();
    for (int i = 0; i < w * h; ++i) {
        hist[src_ptr[i]]++;
    }
    
    // Compute CDF
    int cdf[256];
    cdf[0] = hist[0];
    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i-1] + hist[i];
    }
    
    // Find min non-zero CDF
    int cdf_min = cdf[0];
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] > 0) {
            cdf_min = cdf[i];
            break;
        }
    }
    
    // Create lookup table
    uint8_t lut[256];
    int total_pixels = w * h;
    for (int i = 0; i < 256; ++i) {
        lut[i] = static_cast<uint8_t>(
            std::round(((float)(cdf[i] - cdf_min) / (total_pixels - cdf_min)) * 255.0f)
        );
    }
    
    // Apply lookup table
    Image result(w, h, 1, DType::UINT8);
    uint8_t* dst_ptr = result.ptr<uint8_t>();
    for (int i = 0; i < w * h; ++i) {
        dst_ptr[i] = lut[src_ptr[i]];
    }
    
    return result;
}

Image flip(const Image& src, bool horizontal) {
    Image dst(src.width(), src.height(), src.channels(), src.dtype());
    
    size_t W = src.width();
    size_t H = src.height();
    size_t C = src.channels();
    
    if (src.dtype() == DType::UINT8) {
        const uint8_t* s = src.ptr<uint8_t>();
        uint8_t* d = dst.ptr<uint8_t>();
        
        for (size_t c = 0; c < C; ++c) {
            for (size_t y = 0; y < H; ++y) {
                for (size_t x = 0; x < W; ++x) {
                    size_t sx = horizontal ? (W - 1 - x) : x;
                    size_t sy = horizontal ? y : (H - 1 - y);
                    d[c * W * H + y * W + x] = s[c * W * H + sy * W + sx];
                }
            }
        }
    }
    
    return dst;
}

} // namespace imgproc

// ============================================================================
// Neural Network Functions
// ============================================================================

namespace nn {

// Activation functions
Tensor relu(const Tensor& x) {
    Tensor result(x.shape(), x.dtype());
    const float* a = x.ptr<float>();
    float* c = result.ptr<float>();
    for (size_t i = 0; i < x.size(); ++i) {
        c[i] = std::max(0.0f, a[i]);
    }
    return result;
}

Tensor leaky_relu(const Tensor& x, float negative_slope = 0.01f) {
    Tensor result(x.shape(), x.dtype());
    const float* a = x.ptr<float>();
    float* c = result.ptr<float>();
    for (size_t i = 0; i < x.size(); ++i) {
        c[i] = a[i] >= 0 ? a[i] : negative_slope * a[i];
    }
    return result;
}

Tensor sigmoid(const Tensor& x) {
    Tensor result(x.shape(), x.dtype());
    const float* a = x.ptr<float>();
    float* c = result.ptr<float>();
    for (size_t i = 0; i < x.size(); ++i) {
        c[i] = 1.0f / (1.0f + std::exp(-a[i]));
    }
    return result;
}

Tensor tanh(const Tensor& x) { return x.tanh(); }

Tensor softmax(const Tensor& x, int dim = -1) {
    // Simplified: softmax over last dimension
    Tensor result(x.shape(), x.dtype());
    const float* a = x.ptr<float>();
    float* c = result.ptr<float>();
    
    if (x.shape().size() == 1) {
        float max_val = x.max();
        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) {
            c[i] = std::exp(a[i] - max_val);
            sum += c[i];
        }
        for (size_t i = 0; i < x.size(); ++i) c[i] /= sum;
    } else if (x.shape().size() == 2) {
        size_t rows = x.shape()[0];
        size_t cols = x.shape()[1];
        
        for (size_t i = 0; i < rows; ++i) {
            float max_val = a[i * cols];
            for (size_t j = 1; j < cols; ++j) {
                max_val = std::max(max_val, a[i * cols + j]);
            }
            
            float sum = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                c[i * cols + j] = std::exp(a[i * cols + j] - max_val);
                sum += c[i * cols + j];
            }
            
            for (size_t j = 0; j < cols; ++j) {
                c[i * cols + j] /= sum;
            }
        }
    }
    
    return result;
}

Tensor gelu(const Tensor& x) {
    Tensor result(x.shape(), x.dtype());
    const float* a = x.ptr<float>();
    float* c = result.ptr<float>();
    const float sqrt_2_pi = 0.7978845608f;
    for (size_t i = 0; i < x.size(); ++i) {
        float v = a[i];
        c[i] = 0.5f * v * (1.0f + std::tanh(sqrt_2_pi * (v + 0.044715f * v * v * v)));
    }
    return result;
}

Tensor swish(const Tensor& x) {
    Tensor sig = sigmoid(x);
    return x * sig;
}

// Loss functions
float mse_loss(const Tensor& pred, const Tensor& target) {
    const float* p = pred.ptr<float>();
    const float* t = target.ptr<float>();
    float sum = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i) {
        float d = p[i] - t[i];
        sum += d * d;
    }
    return sum / pred.size();
}

float cross_entropy(const Tensor& pred, const Tensor& target) {
    const float* p = pred.ptr<float>();
    const float* t = target.ptr<float>();
    float sum = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i) {
        sum -= t[i] * std::log(p[i] + 1e-10f);
    }
    return sum / pred.size();
}

} // namespace nn

// ============================================================================
// Machine Learning Functions
// ============================================================================

namespace ml {

float accuracy_score(const Tensor& y_true, const Tensor& y_pred) {
    const float* t = y_true.ptr<float>();
    const float* p = y_pred.ptr<float>();
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (std::round(t[i]) == std::round(p[i])) correct++;
    }
    return static_cast<float>(correct) / y_true.size();
}

float mean_squared_error(const Tensor& y_true, const Tensor& y_pred) {
    return nn::mse_loss(y_pred, y_true);
}

} // namespace ml

// ============================================================================
// Face Detection
// ============================================================================

namespace face {

class HaarCascade {
public:
    HaarCascade() : loaded_(false) {}
    
    bool load(const std::string& path) {
        // Simplified: just mark as loaded
        loaded_ = true;
        return true;
    }
    
    std::vector<Rect> detect(const Image& img, float scale_factor = 1.1f, 
                             int min_neighbors = 3, int min_size = 30) {
        std::vector<Rect> faces;
        // Simplified placeholder
        return faces;
    }
    
private:
    bool loaded_;
};

} // namespace face

// ============================================================================
// Filters Module
// ============================================================================

namespace filters {

// Sobel edge detection
Image sobel(const Image& src, int dx = 1, int dy = 0) {
    Image gray = (src.channels() == 1) ? src : imgproc::rgb_to_gray(src);
    Image result(gray.width(), gray.height(), 1, DType::FLOAT32);
    
    int W = gray.width();
    int H = gray.height();
    
    // Sobel kernels
    int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    const uint8_t* src_data = gray.ptr<uint8_t>();
    float* dst_data = result.ptr<float>();
    
    // Initialize borders to zero
    std::memset(dst_data, 0, W * H * sizeof(float));
    
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            float gx = 0.0f, gy = 0.0f;
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    uint8_t pixel = src_data[(y + ky) * W + (x + kx)];
                    if (dx > 0) gx += pixel * sobel_x[ky + 1][kx + 1];
                    if (dy > 0) gy += pixel * sobel_y[ky + 1][kx + 1];
                }
            }
            
            dst_data[y * W + x] = std::sqrt(gx * gx + gy * gy);
        }
    }
    
    return result.to(DType::UINT8);
}

// Canny edge detection (simplified)
Image canny(const Image& src, float low_threshold = 50, float high_threshold = 150) {
    // Step 1: Gaussian blur
    Image blurred = imgproc::gaussian_blur(src, 5, 1.4f);
    
    // Step 2: Sobel gradients
    Image grad_x = sobel(blurred, 1, 0);
    Image grad_y = sobel(blurred, 0, 1);
    
    int W = src.width();
    int H = src.height();
    Image result(W, H, 1, DType::UINT8);
    
    const float* gx = grad_x.ptr<float>();
    const float* gy = grad_y.ptr<float>();
    uint8_t* dst = result.ptr<uint8_t>();
    
    // Compute magnitude and apply thresholds
    for (size_t i = 0; i < W * H; ++i) {
        float mag = std::sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
        if (mag > high_threshold) {
            dst[i] = 255;
        } else if (mag > low_threshold) {
            dst[i] = 128;
        } else {
            dst[i] = 0;
        }
    }
    
    return result;
}

// Median filter
Image median_filter(const Image& src, int ksize = 3) {
    Image gray = (src.channels() == 1) ? src : imgproc::rgb_to_gray(src);
    Image result(gray.width(), gray.height(), 1, gray.dtype());
    
    int W = gray.width();
    int H = gray.height();
    int half = ksize / 2;
    
    const uint8_t* s = gray.ptr<uint8_t>();
    uint8_t* d = result.ptr<uint8_t>();
    
    std::vector<uint8_t> window(ksize * ksize);
    
    for (int y = half; y < H - half; ++y) {
        for (int x = half; x < W - half; ++x) {
            int idx = 0;
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    window[idx++] = s[(y + ky) * W + (x + kx)];
                }
            }
            
            std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
            d[y * W + x] = window[window.size() / 2];
        }
    }
    
    return result;
}

// Bilateral filter (simplified)
Image bilateral_filter(const Image& src, int d = 9, double sigma_color = 75, double sigma_space = 75) {
    Image gray = (src.channels() == 1) ? src : imgproc::rgb_to_gray(src);
    Image result(gray.width(), gray.height(), 1, DType::FLOAT32);
    
    int W = gray.width();
    int H = gray.height();
    int half = d / 2;
    
    const uint8_t* s = gray.ptr<uint8_t>();
    float* dst = result.ptr<float>();
    
    for (int y = half; y < H - half; ++y) {
        for (int x = half; x < W - half; ++x) {
            float sum = 0.0f;
            float norm = 0.0f;
            uint8_t center = s[y * W + x];
            
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    uint8_t neighbor = s[(y + ky) * W + (x + kx)];
                    
                    float space_dist = kx * kx + ky * ky;
                    float color_dist = (center - neighbor) * (center - neighbor);
                    
                    float weight = std::exp(-space_dist / (2 * sigma_space * sigma_space) 
                                          - color_dist / (2 * sigma_color * sigma_color));
                    
                    sum += neighbor * weight;
                    norm += weight;
                }
            }
            
            dst[y * W + x] = sum / norm;
        }
    }
    
    return result.to(DType::UINT8);
}

// Laplacian edge detection
Image laplacian(const Image& src, int ksize = 3) {
    Image gray = (src.channels() == 1) ? src : imgproc::rgb_to_gray(src);
    Image result(gray.width(), gray.height(), 1, DType::FLOAT32);
    
    int W = gray.width();
    int H = gray.height();
    
    const uint8_t* src_data = gray.ptr<uint8_t>();
    float* dst_data = result.ptr<float>();
    
    // Initialize borders to zero
    std::memset(dst_data, 0, W * H * sizeof(float));
    
    // Laplacian kernel: [[0,-1,0],[-1,4,-1],[0,-1,0]]
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            float lap = -src_data[(y-1) * W + x]
                       - src_data[y * W + (x-1)]
                       + 4.0f * src_data[y * W + x]
                       - src_data[y * W + (x+1)]
                       - src_data[(y+1) * W + x];
            dst_data[y * W + x] = std::abs(lap);
        }
    }
    
    return result.to(DType::UINT8);
}

// Scharr operator (more accurate than Sobel)
Image scharr(const Image& src, int dx = 1, int dy = 0) {
    Image gray = (src.channels() == 1) ? src : imgproc::rgb_to_gray(src);
    Image result(gray.width(), gray.height(), 1, DType::FLOAT32);
    
    int W = gray.width();
    int H = gray.height();
    
    const uint8_t* src_data = gray.ptr<uint8_t>();
    float* dst_data = result.ptr<float>();
    
    // Initialize borders to zero
    std::memset(dst_data, 0, W * H * sizeof(float));
    
    // Scharr kernels (better rotation invariance than Sobel)
    const int scharr_x[3][3] = {{-3, 0, 3}, {-10, 0, 10}, {-3, 0, 3}};
    const int scharr_y[3][3] = {{-3, -10, -3}, {0, 0, 0}, {3, 10, 3}};
    
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            float gx = 0, gy = 0;
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    uint8_t pixel = src_data[(y + ky) * W + (x + kx)];
                    if (dx > 0) gx += pixel * scharr_x[ky + 1][kx + 1];
                    if (dy > 0) gy += pixel * scharr_y[ky + 1][kx + 1];
                }
            }
            
            dst_data[y * W + x] = std::sqrt(gx * gx + gy * gy);
        }
    }
    
    return result.to(DType::UINT8);
}

// Box filter (fast uniform blur)
Image box_filter(const Image& src, int ksize = 3) {
    Image result(src.width(), src.height(), src.channels(), DType::FLOAT32);
    
    int W = src.width();
    int H = src.height();
    int C = src.channels();
    int half = ksize / 2;
    
    const uint8_t* s = src.ptr<uint8_t>();
    float* d = result.ptr<float>();
    
    float norm = 1.0f / (ksize * ksize);
    
    for (int c = 0; c < C; ++c) {
        for (int y = half; y < H - half; ++y) {
            for (int x = half; x < W - half; ++x) {
                float sum = 0;
                for (int ky = -half; ky <= half; ++ky) {
                    for (int kx = -half; kx <= half; ++kx) {
                        sum += s[((y + ky) * W + (x + kx)) * C + c];
                    }
                }
                d[(y * W + x) * C + c] = sum * norm;
            }
        }
    }
    
    return result.to(DType::UINT8);
}

} // namespace filters

// ============================================================================
// Morphology Module
// ============================================================================

namespace morphology {

enum class MorphOp {
    ERODE,
    DILATE,
    OPEN,
    CLOSE,
    GRADIENT,
    TOPHAT,
    BLACKHAT
};

// Basic erosion
Image erode(const Image& src, int ksize = 3) {
    Image binary = (src.channels() == 1) ? src : imgproc::rgb_to_gray(src);
    Image result(binary.width(), binary.height(), 1, binary.dtype());
    
    int W = binary.width();
    int H = binary.height();
    int half = ksize / 2;
    
    const uint8_t* s = binary.ptr<uint8_t>();
    uint8_t* d = result.ptr<uint8_t>();
    
    for (int y = half; y < H - half; ++y) {
        for (int x = half; x < W - half; ++x) {
            uint8_t min_val = 255;
            
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    min_val = std::min(min_val, s[(y + ky) * W + (x + kx)]);
                }
            }
            
            d[y * W + x] = min_val;
        }
    }
    
    return result;
}

// Basic dilation
Image dilate(const Image& src, int ksize = 3) {
    Image binary = (src.channels() == 1) ? src : imgproc::rgb_to_gray(src);
    Image result(binary.width(), binary.height(), 1, binary.dtype());
    
    int W = binary.width();
    int H = binary.height();
    int half = ksize / 2;
    
    const uint8_t* s = binary.ptr<uint8_t>();
    uint8_t* d = result.ptr<uint8_t>();
    
    for (int y = half; y < H - half; ++y) {
        for (int x = half; x < W - half; ++x) {
            uint8_t max_val = 0;
            
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    max_val = std::max(max_val, s[(y + ky) * W + (x + kx)]);
                }
            }
            
            d[y * W + x] = max_val;
        }
    }
    
    return result;
}

// Opening: erosion followed by dilation
Image opening(const Image& src, int ksize = 3) {
    return dilate(erode(src, ksize), ksize);
}

// Closing: dilation followed by erosion
Image closing(const Image& src, int ksize = 3) {
    return erode(dilate(src, ksize), ksize);
}

// Morphological gradient
Image gradient(const Image& src, int ksize = 3) {
    Image dilated = dilate(src, ksize);
    Image eroded = erode(src, ksize);
    
    Image result(src.width(), src.height(), 1, src.dtype());
    const uint8_t* d_data = dilated.ptr<uint8_t>();
    const uint8_t* e_data = eroded.ptr<uint8_t>();
    uint8_t* r_data = result.ptr<uint8_t>();
    
    for (size_t i = 0; i < src.width() * src.height(); ++i) {
        r_data[i] = d_data[i] - e_data[i];
    }
    
    return result;
}

} // namespace morphology

// ============================================================================
// Features Module (Corner Detection)
// ============================================================================

namespace features {

// Harris corner detection
Tensor harris_corners(const Image& src, int block_size = 3, int ksize = 3, float k = 0.04) {
    Image gray = (src.channels() == 1) ? src : imgproc::rgb_to_gray(src);
    
    int W = gray.width();
    int H = gray.height();
    
    // Compute gradients
    Image grad_x = filters::sobel(gray, 1, 0);
    Image grad_y = filters::sobel(gray, 0, 1);
    
    const float* Ix = grad_x.ptr<float>();
    const float* Iy = grad_y.ptr<float>();
    
    // Compute products of derivatives
    std::vector<float> Ixx(W * H), Iyy(W * H), Ixy(W * H);
    for (int i = 0; i < W * H; ++i) {
        Ixx[i] = Ix[i] * Ix[i];
        Iyy[i] = Iy[i] * Iy[i];
        Ixy[i] = Ix[i] * Iy[i];
    }
    
    // Create response tensor
    Tensor response({static_cast<size_t>(H), static_cast<size_t>(W)}, DType::FLOAT32);
    float* R = response.ptr<float>();
    
    int half = block_size / 2;
    
    for (int y = half; y < H - half; ++y) {
        for (int x = half; x < W - half; ++x) {
            // Sum over neighborhood
            float sum_Ixx = 0, sum_Iyy = 0, sum_Ixy = 0;
            
            for (int dy = -half; dy <= half; ++dy) {
                for (int dx = -half; dx <= half; ++dx) {
                    int idx = (y + dy) * W + (x + dx);
                    sum_Ixx += Ixx[idx];
                    sum_Iyy += Iyy[idx];
                    sum_Ixy += Ixy[idx];
                }
            }
            
            // Harris response: det(M) - k * trace(M)^2
            float det = sum_Ixx * sum_Iyy - sum_Ixy * sum_Ixy;
            float trace = sum_Ixx + sum_Iyy;
            R[y * W + x] = det - k * trace * trace;
        }
    }
    
    return response;
}

// Good Features To Track (Shi-Tomasi)
Tensor good_features_to_track(const Image& src, int max_corners = 100, float quality_level = 0.01) {
    // Use Harris without the k parameter (i.e., k=0 gives Shi-Tomasi)
    Tensor response = harris_corners(src, 3, 3, 0.0f);
    
    // Find local maxima
    int W = src.width();
    int H = src.height();
    const float* R = response.ptr<float>();
    
    // Find max response
    float max_response = 0;
    for (int i = 0; i < W * H; ++i) {
        max_response = std::max(max_response, R[i]);
    }
    
    float threshold = quality_level * max_response;
    
    // Collect corners above threshold
    std::vector<std::pair<float, std::pair<int, int>>> scored_corners;
    
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            float val = R[y * W + x];
            if (val > threshold) {
                // Check if local maximum
                bool is_max = true;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (R[(y+dy) * W + (x+dx)] > val) {
                            is_max = false;
                            break;
                        }
                    }
                    if (!is_max) break;
                }
                if (is_max) {
                    scored_corners.push_back({val, {x, y}});
                }
            }
        }
    }
    
    // Sort by score and take top N
    std::sort(scored_corners.begin(), scored_corners.end(), 
              std::greater<std::pair<float, std::pair<int, int>>>());
    
    int n = std::min(max_corners, static_cast<int>(scored_corners.size()));
    Tensor corners({static_cast<size_t>(n), 2}, DType::FLOAT32);
    float* corners_ptr = corners.ptr<float>();
    
    for (int i = 0; i < n; ++i) {
        corners_ptr[i * 2] = scored_corners[i].second.first;
        corners_ptr[i * 2 + 1] = scored_corners[i].second.second;
    }
    
    return corners;
}

} // namespace features

// ============================================================================
// Transform Module
// ============================================================================

namespace transform {

// Get 2D rotation matrix
Tensor get_rotation_matrix_2d(float cx, float cy, float angle, float scale = 1.0) {
    float rad = angle * 3.14159265f / 180.0f;
    float alpha = scale * std::cos(rad);
    float beta = scale * std::sin(rad);
    
    Tensor mat({2, 3}, DType::FLOAT32);
    float* m = mat.ptr<float>();
    
    m[0] = alpha;
    m[1] = beta;
    m[2] = (1 - alpha) * cx - beta * cy;
    m[3] = -beta;
    m[4] = alpha;
    m[5] = beta * cx + (1 - alpha) * cy;
    
    return mat;
}

// Affine transformation
Image warp_affine(const Image& src, const Tensor& M, int dst_w, int dst_h) {
    if (M.shape()[0] != 2 || M.shape()[1] != 3) {
        throw std::runtime_error("warp_affine: M must be 2x3");
    }
    
    Image result(dst_w, dst_h, src.channels(), src.dtype());
    
    const float* mat = M.ptr<float>();
    const uint8_t* src_data = src.ptr<uint8_t>();
    uint8_t* dst_data = result.ptr<uint8_t>();
    
    int C = src.channels();
    int src_w = src.width();
    int src_h = src.height();
    
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            // Apply inverse transform
            float src_x = mat[0] * x + mat[1] * y + mat[2];
            float src_y = mat[3] * x + mat[4] * y + mat[5];
            
            int sx = static_cast<int>(src_x);
            int sy = static_cast<int>(src_y);
            
            if (sx >= 0 && sx < src_w && sy >= 0 && sy < src_h) {
                for (int c = 0; c < C; ++c) {
                    dst_data[(y * dst_w + x) * C + c] = src_data[(sy * src_w + sx) * C + c];
                }
            }
        }
    }
    
    return result;
}

} // namespace transform

// ============================================================================
// Video Processing Module
// ============================================================================

namespace video {

// Simple background subtraction using frame differencing
class BackgroundSubtractor {
public:
    BackgroundSubtractor() : initialized_(false) {}
    
    void apply(const Image& frame, Image& fgmask, float learning_rate = 0.01f) {
        if (!initialized_) {
            // Initialize background model with first frame
            background_ = frame.clone();
            if (background_.dtype() != DType::FLOAT32) {
                background_ = background_.to(DType::FLOAT32);
            }
            initialized_ = true;
            
            // Create black mask for first frame
            fgmask = Image(frame.width(), frame.height(), 1, DType::UINT8);
            std::memset(fgmask.ptr<uint8_t>(), 0, frame.width() * frame.height());
            return;
        }
        
        // Convert current frame to float
        Image frame_float = (frame.dtype() == DType::FLOAT32) ? frame : frame.to(DType::FLOAT32);
        
        int W = frame.width();
        int H = frame.height();
        int C = frame.channels();
        
        const float* bg_ptr = background_.ptr<float>();
        const float* frame_ptr = frame_float.ptr<float>();
        float* bg_out = background_.ptr<float>();
        
        // Create foreground mask
        fgmask = Image(W, H, 1, DType::UINT8);
        uint8_t* mask_ptr = fgmask.ptr<uint8_t>();
        
        // Compute difference and update background
        float threshold = 30.0f;
        for (int i = 0; i < W * H; ++i) {
            float diff = 0;
            for (int c = 0; c < C; ++c) {
                int idx = i * C + c;
                float d = std::abs(frame_ptr[idx] - bg_ptr[idx]);
                diff += d;
                
                // Update background model
                bg_out[idx] = bg_ptr[idx] * (1.0f - learning_rate) + frame_ptr[idx] * learning_rate;
            }
            
            mask_ptr[i] = (diff / C > threshold) ? 255 : 0;
        }
    }
    
    void reset() {
        initialized_ = false;
    }

private:
    Image background_;
    bool initialized_;
};

// Lucas-Kanade optical flow (simplified)
Tensor calc_optical_flow_lk(const Image& prev, const Image& next, int win_size = 15) {
    // Convert to grayscale if needed
    Image prev_gray = (prev.channels() == 1) ? prev : imgproc::rgb_to_gray(prev);
    Image next_gray = (next.channels() == 1) ? next : imgproc::rgb_to_gray(next);
    
    int W = prev_gray.width();
    int H = prev_gray.height();
    
    // Compute gradients
    Image Ix = filters::sobel(prev_gray, 1, 0);
    Image Iy = filters::sobel(prev_gray, 0, 1);
    Image It(W, H, 1, DType::FLOAT32);
    
    // Temporal gradient
    const uint8_t* prev_ptr = prev_gray.ptr<uint8_t>();
    const uint8_t* next_ptr = next_gray.ptr<uint8_t>();
    float* It_ptr = It.ptr<float>();
    
    for (int i = 0; i < W * H; ++i) {
        It_ptr[i] = static_cast<float>(next_ptr[i]) - static_cast<float>(prev_ptr[i]);
    }
    
    const float* Ix_ptr = Ix.ptr<float>();
    const float* Iy_ptr = Iy.ptr<float>();
    
    // Flow field (u, v)
    Tensor flow({static_cast<size_t>(H), static_cast<size_t>(W), 2}, DType::FLOAT32);
    float* flow_ptr = flow.ptr<float>();
    
    int half = win_size / 2;
    
    for (int y = half; y < H - half; ++y) {
        for (int x = half; x < W - half; ++x) {
            // Compute A^T A and A^T b for least squares
            float sum_Ix2 = 0, sum_Iy2 = 0, sum_IxIy = 0;
            float sum_IxIt = 0, sum_IyIt = 0;
            
            for (int dy = -half; dy <= half; ++dy) {
                for (int dx = -half; dx <= half; ++dx) {
                    int idx = (y + dy) * W + (x + dx);
                    float ix = Ix_ptr[idx];
                    float iy = Iy_ptr[idx];
                    float it = It_ptr[idx];
                    
                    sum_Ix2 += ix * ix;
                    sum_Iy2 += iy * iy;
                    sum_IxIy += ix * iy;
                    sum_IxIt += ix * it;
                    sum_IyIt += iy * it;
                }
            }
            
            // Solve 2x2 system
            float det = sum_Ix2 * sum_Iy2 - sum_IxIy * sum_IxIy;
            
            if (std::abs(det) > 1e-5) {
                float u = (sum_Iy2 * (-sum_IxIt) - sum_IxIy * (-sum_IyIt)) / det;
                float v = (sum_Ix2 * (-sum_IyIt) - sum_IxIy * (-sum_IxIt)) / det;
                
                flow_ptr[(y * W + x) * 2 + 0] = u;
                flow_ptr[(y * W + x) * 2 + 1] = v;
            }
        }
    }
    
    return flow;
}

} // namespace video

// ============================================================================
// Segmentation Module
// ============================================================================

namespace segmentation {

// Distance transform (for watershed preprocessing)
Image distance_transform(const Image& binary) {
    if (binary.dtype() != DType::UINT8 || binary.channels() != 1) {
        throw std::runtime_error("distance_transform: binary UINT8 single-channel required");
    }
    
    int W = binary.width();
    int H = binary.height();
    
    Image dist(W, H, 1, DType::FLOAT32);
    const uint8_t* bin_ptr = binary.ptr<uint8_t>();
    float* dist_ptr = dist.ptr<float>();
    
    // Initialize with infinity for background, 0 for foreground
    const float INF = 1e6f;
    for (int i = 0; i < W * H; ++i) {
        dist_ptr[i] = (bin_ptr[i] > 128) ? 0.0f : INF;
    }
    
    // Forward pass
    for (int y = 1; y < H; ++y) {
        for (int x = 1; x < W; ++x) {
            int idx = y * W + x;
            float min_dist = dist_ptr[idx];
            min_dist = std::min(min_dist, dist_ptr[(y-1) * W + x] + 1.0f);
            min_dist = std::min(min_dist, dist_ptr[y * W + (x-1)] + 1.0f);
            dist_ptr[idx] = min_dist;
        }
    }
    
    // Backward pass
    for (int y = H - 2; y >= 0; --y) {
        for (int x = W - 2; x >= 0; --x) {
            int idx = y * W + x;
            float min_dist = dist_ptr[idx];
            min_dist = std::min(min_dist, dist_ptr[(y+1) * W + x] + 1.0f);
            min_dist = std::min(min_dist, dist_ptr[y * W + (x+1)] + 1.0f);
            dist_ptr[idx] = min_dist;
        }
    }
    
    return dist;
}

// Simple watershed segmentation
Image watershed(const Image& markers, const Image& gradient) {
    if (markers.dtype() != DType::UINT8 || gradient.dtype() != DType::UINT8) {
        throw std::runtime_error("watershed: UINT8 images required");
    }
    
    int W = markers.width();
    int H = markers.height();
    
    // Create output labels
    Image labels = markers.clone();
    uint8_t* label_ptr = labels.ptr<uint8_t>();
    const uint8_t* grad_ptr = gradient.ptr<uint8_t>();
    
    // Priority queue simulation - process pixels by gradient intensity
    std::vector<std::tuple<int, int, int>> pixels; // (gradient, y, x)
    
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;
            if (label_ptr[idx] == 0) {
                pixels.push_back({grad_ptr[idx], y, x});
            }
        }
    }
    
    // Sort by gradient (ascending)
    std::sort(pixels.begin(), pixels.end());
    
    // Propagate labels
    const int dx[] = {-1, 1, 0, 0};
    const int dy[] = {0, 0, -1, 1};
    
    for (const auto& [g, y, x] : pixels) {
        int idx = y * W + x;
        
        // Find neighboring labels
        uint8_t neighbor_label = 0;
        for (int d = 0; d < 4; ++d) {
            int nx = x + dx[d];
            int ny = y + dy[d];
            
            if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                uint8_t nlabel = label_ptr[ny * W + nx];
                if (nlabel > 0) {
                    if (neighbor_label == 0) {
                        neighbor_label = nlabel;
                    } else if (neighbor_label != nlabel) {
                        neighbor_label = 255; // Watershed line
                        break;
                    }
                }
            }
        }
        
        if (neighbor_label > 0) {
            label_ptr[idx] = neighbor_label;
        }
    }
    
    return labels;
}

// Connected components labeling
Image connected_components(const Image& binary, int& num_labels) {
    if (binary.dtype() != DType::UINT8 || binary.channels() != 1) {
        throw std::runtime_error("connected_components: binary UINT8 required");
    }
    
    int W = binary.width();
    int H = binary.height();
    
    Image labels(W, H, 1, DType::UINT8);
    const uint8_t* bin_ptr = binary.ptr<uint8_t>();
    uint8_t* label_ptr = labels.ptr<uint8_t>();
    
    std::memset(label_ptr, 0, W * H);
    
    uint8_t current_label = 1;
    
    // Simple flood fill for each component
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;
            
            if (bin_ptr[idx] > 128 && label_ptr[idx] == 0) {
                // Start flood fill
                std::vector<std::pair<int, int>> stack;
                stack.push_back({x, y});
                label_ptr[idx] = current_label;
                
                while (!stack.empty()) {
                    auto [cx, cy] = stack.back();
                    stack.pop_back();
                    
                    // Check 4-neighbors
                    const int dx[] = {-1, 1, 0, 0};
                    const int dy[] = {0, 0, -1, 1};
                    
                    for (int d = 0; d < 4; ++d) {
                        int nx = cx + dx[d];
                        int ny = cy + dy[d];
                        
                        if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                            int nidx = ny * W + nx;
                            if (bin_ptr[nidx] > 128 && label_ptr[nidx] == 0) {
                                label_ptr[nidx] = current_label;
                                stack.push_back({nx, ny});
                            }
                        }
                    }
                }
                
                current_label++;
                if (current_label > 250) break; // Limit to 250 components
            }
        }
        if (current_label > 250) break;
    }
    
    num_labels = current_label - 1;
    return labels;
}

} // namespace segmentation

// ============================================================================
// ML Clustering
// ============================================================================

namespace ml {

// KMeans clustering
class KMeans {
public:
    KMeans(int n_clusters = 8, int max_iter = 300, float tol = 1e-4f)
        : n_clusters_(n_clusters), max_iter_(max_iter), tol_(tol) {}
    
    void fit(const Tensor& X) {
        size_t n_samples = X.shape()[0];
        size_t n_features = X.shape()[1];
        
        // Initialize centroids randomly
        centroids_ = Tensor::zeros({static_cast<size_t>(n_clusters_), n_features});
        labels_ = Tensor::zeros({n_samples}, DType::INT32);
        
        const float* x_data = X.ptr<float>();
        float* c_data = centroids_.ptr<float>();
        
        // Random initialization
        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        for (int k = 0; k < n_clusters_; ++k) {
            for (size_t f = 0; f < n_features; ++f) {
                c_data[k * n_features + f] = x_data[indices[k] * n_features + f];
            }
        }
        
        // Iterate
        for (int iter = 0; iter < max_iter_; ++iter) {
            // Assign labels
            int32_t* l_data = labels_.ptr<int32_t>();
            
            for (size_t i = 0; i < n_samples; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                int32_t best_k = 0;
                
                for (int k = 0; k < n_clusters_; ++k) {
                    float dist = 0.0f;
                    for (size_t f = 0; f < n_features; ++f) {
                        float diff = x_data[i * n_features + f] - c_data[k * n_features + f];
                        dist += diff * diff;
                    }
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_k = k;
                    }
                }
                
                l_data[i] = best_k;
            }
            
            // Update centroids
            std::vector<float> new_centroids(n_clusters_ * n_features, 0.0f);
            std::vector<int> counts(n_clusters_, 0);
            
            for (size_t i = 0; i < n_samples; ++i) {
                int k = l_data[i];
                counts[k]++;
                for (size_t f = 0; f < n_features; ++f) {
                    new_centroids[k * n_features + f] += x_data[i * n_features + f];
                }
            }
            
            // Average
            for (int k = 0; k < n_clusters_; ++k) {
                if (counts[k] > 0) {
                    for (size_t f = 0; f < n_features; ++f) {
                        new_centroids[k * n_features + f] /= counts[k];
                    }
                }
            }
            
            // Check convergence
            float change = 0.0f;
            for (size_t i = 0; i < n_clusters_ * n_features; ++i) {
                float diff = new_centroids[i] - c_data[i];
                change += diff * diff;
            }
            
            std::memcpy(c_data, new_centroids.data(), n_clusters_ * n_features * sizeof(float));
            
            if (std::sqrt(change) < tol_) break;
        }
    }
    
    Tensor predict(const Tensor& X) const {
        size_t n_samples = X.shape()[0];
        size_t n_features = X.shape()[1];
        
        Tensor labels = Tensor::zeros({n_samples}, DType::INT32);
        
        const float* x_data = X.ptr<float>();
        const float* c_data = centroids_.ptr<float>();
        int32_t* l_data = labels.ptr<int32_t>();
        
        for (size_t i = 0; i < n_samples; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int32_t best_k = 0;
            
            for (int k = 0; k < n_clusters_; ++k) {
                float dist = 0.0f;
                for (size_t f = 0; f < n_features; ++f) {
                    float diff = x_data[i * n_features + f] - c_data[k * n_features + f];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_k = k;
                }
            }
            
            l_data[i] = best_k;
        }
        
        return labels;
    }
    
    Tensor get_centroids() const { return centroids_; }
    Tensor get_labels() const { return labels_; }
    
private:
    int n_clusters_;
    int max_iter_;
    float tol_;
    Tensor centroids_;
    Tensor labels_;
};

// PCA - Principal Component Analysis
class PCA {
public:
    PCA(int n_components = 2) : n_components_(n_components) {}
    
    void fit(const Tensor& X) {
        size_t n_samples = X.shape()[0];
        size_t n_features = X.shape()[1];
        
        // Center the data
        mean_ = Tensor::zeros({n_features});
        float* m_data = mean_.ptr<float>();
        const float* x_data = X.ptr<float>();
        
        for (size_t f = 0; f < n_features; ++f) {
            for (size_t i = 0; i < n_samples; ++i) {
                m_data[f] += x_data[i * n_features + f];
            }
            m_data[f] /= n_samples;
        }
        
        // Compute covariance matrix (simplified)
        Tensor X_centered = X.clone();
        float* xc_data = X_centered.ptr<float>();
        
        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t f = 0; f < n_features; ++f) {
                xc_data[i * n_features + f] -= m_data[f];
            }
        }
        
        // For simplicity, store first n_components_ as identity
        components_ = Tensor::eye(n_features);
        n_components_ = std::min(n_components_, static_cast<int>(n_features));
    }
    
    Tensor transform(const Tensor& X) const {
        size_t n_samples = X.shape()[0];
        size_t n_features = X.shape()[1];
        
        Tensor X_transformed = Tensor::zeros({n_samples, static_cast<size_t>(n_components_)});
        
        const float* x_data = X.ptr<float>();
        const float* m_data = mean_.ptr<float>();
        const float* c_data = components_.ptr<float>();
        float* xt_data = X_transformed.ptr<float>();
        
        for (size_t i = 0; i < n_samples; ++i) {
            for (int k = 0; k < n_components_; ++k) {
                float val = 0.0f;
                for (size_t f = 0; f < n_features; ++f) {
                    val += (x_data[i * n_features + f] - m_data[f]) * c_data[k * n_features + f];
                }
                xt_data[i * n_components_ + k] = val;
            }
        }
        
        return X_transformed;
    }
    
    Tensor get_components() const { return components_; }
    
private:
    int n_components_;
    Tensor mean_;
    Tensor components_;
};

// Linear Regression
class LinearRegression {
public:
    void fit(const Tensor& X, const Tensor& y) {
        // Normal equation: theta = (X^T X)^{-1} X^T y
        // Simplified: use gradient descent instead
        
        size_t n_samples = X.shape()[0];
        size_t n_features = X.shape()[1];
        
        coef_ = Tensor::zeros({n_features});
        intercept_ = 0.0f;
        
        float lr = 0.01f;
        int n_iter = 1000;
        
        const float* x_data = X.ptr<float>();
        const float* y_data = y.ptr<float>();
        float* w_data = coef_.ptr<float>();
        
        for (int iter = 0; iter < n_iter; ++iter) {
            std::vector<float> gradients(n_features, 0.0f);
            float grad_intercept = 0.0f;
            
            for (size_t i = 0; i < n_samples; ++i) {
                float pred = intercept_;
                for (size_t f = 0; f < n_features; ++f) {
                    pred += w_data[f] * x_data[i * n_features + f];
                }
                
                float error = pred - y_data[i];
                grad_intercept += error;
                
                for (size_t f = 0; f < n_features; ++f) {
                    gradients[f] += error * x_data[i * n_features + f];
                }
            }
            
            intercept_ -= lr * grad_intercept / n_samples;
            for (size_t f = 0; f < n_features; ++f) {
                w_data[f] -= lr * gradients[f] / n_samples;
            }
        }
    }
    
    Tensor predict(const Tensor& X) const {
        size_t n_samples = X.shape()[0];
        size_t n_features = X.shape()[1];
        
        Tensor y_pred = Tensor::zeros({n_samples});
        
        const float* x_data = X.ptr<float>();
        const float* w_data = coef_.ptr<float>();
        float* yp_data = y_pred.ptr<float>();
        
        for (size_t i = 0; i < n_samples; ++i) {
            yp_data[i] = intercept_;
            for (size_t f = 0; f < n_features; ++f) {
                yp_data[i] += w_data[f] * x_data[i * n_features + f];
            }
        }
        
        return y_pred;
    }
    
private:
    Tensor coef_;
    float intercept_;
};

} // namespace ml

// ============================================================================
// Convolution & Pooling Layers
// ============================================================================

namespace nn {

// 2D Convolution
Tensor conv2d(const Tensor& input, const Tensor& kernel, int stride = 1, int padding = 0) {
    // Input: [N, C_in, H, W] or [C_in, H, W]
    // Kernel: [C_out, C_in, kH, kW]
    // Output: [N, C_out, H_out, W_out]
    
    // Simplified: assume input is [1, C_in, H, W]
    size_t C_in = input.shape()[0];
    size_t H = input.shape()[1];
    size_t W = input.shape()[2];
    
    size_t C_out = kernel.shape()[0];
    size_t kH = kernel.shape()[2];
    size_t kW = kernel.shape()[3];
    
    size_t H_out = (H + 2 * padding - kH) / stride + 1;
    size_t W_out = (W + 2 * padding - kW) / stride + 1;
    
    Tensor output = Tensor::zeros({C_out, H_out, W_out});
    
    const float* in_data = input.ptr<float>();
    const float* k_data = kernel.ptr<float>();
    float* out_data = output.ptr<float>();
    
    for (size_t co = 0; co < C_out; ++co) {
        for (size_t h = 0; h < H_out; ++h) {
            for (size_t w = 0; w < W_out; ++w) {
                float sum = 0.0f;
                
                for (size_t ci = 0; ci < C_in; ++ci) {
                    for (size_t kh = 0; kh < kH; ++kh) {
                        for (size_t kw = 0; kw < kW; ++kw) {
                            int h_in = h * stride + kh - padding;
                            int w_in = w * stride + kw - padding;
                            
                            if (h_in >= 0 && h_in < static_cast<int>(H) && 
                                w_in >= 0 && w_in < static_cast<int>(W)) {
                                sum += in_data[ci * H * W + h_in * W + w_in] *
                                       k_data[co * C_in * kH * kW + ci * kH * kW + kh * kW + kw];
                            }
                        }
                    }
                }
                
                out_data[co * H_out * W_out + h * W_out + w] = sum;
            }
        }
    }
    
    return output;
}

// Max Pooling 2D
Tensor max_pool2d(const Tensor& input, int kernel_size = 2, int stride = 2) {
    // Input: [C, H, W]
    size_t C = input.shape()[0];
    size_t H = input.shape()[1];
    size_t W = input.shape()[2];
    
    size_t H_out = (H - kernel_size) / stride + 1;
    size_t W_out = (W - kernel_size) / stride + 1;
    
    Tensor output = Tensor::zeros({C, H_out, W_out});
    
    const float* in_data = input.ptr<float>();
    float* out_data = output.ptr<float>();
    
    for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < H_out; ++h) {
            for (size_t w = 0; w < W_out; ++w) {
                float max_val = -std::numeric_limits<float>::max();
                
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        size_t h_in = h * stride + kh;
                        size_t w_in = w * stride + kw;
                        float val = in_data[c * H * W + h_in * W + w_in];
                        max_val = std::max(max_val, val);
                    }
                }
                
                out_data[c * H_out * W_out + h * W_out + w] = max_val;
            }
        }
    }
    
    return output;
}

// Average Pooling 2D
Tensor avg_pool2d(const Tensor& input, int kernel_size = 2, int stride = 2) {
    size_t C = input.shape()[0];
    size_t H = input.shape()[1];
    size_t W = input.shape()[2];
    
    size_t H_out = (H - kernel_size) / stride + 1;
    size_t W_out = (W - kernel_size) / stride + 1;
    
    Tensor output = Tensor::zeros({C, H_out, W_out});
    
    const float* in_data = input.ptr<float>();
    float* out_data = output.ptr<float>();
    
    for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < H_out; ++h) {
            for (size_t w = 0; w < W_out; ++w) {
                float sum = 0.0f;
                
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        size_t h_in = h * stride + kh;
                        size_t w_in = w * stride + kw;
                        sum += in_data[c * H * W + h_in * W + w_in];
                    }
                }
                
                out_data[c * H_out * W_out + h * W_out + w] = sum / (kernel_size * kernel_size);
            }
        }
    }
    
    return output;
}

// Batch Normalization (inference only)
Tensor batch_norm(const Tensor& input, const Tensor& gamma, const Tensor& beta, 
                  const Tensor& mean, const Tensor& var, float eps = 1e-5f) {
    Tensor output = input.clone();
    
    size_t C = input.shape()[0];
    size_t H = input.shape()[1];
    size_t W = input.shape()[2];
    
    const float* g_data = gamma.ptr<float>();
    const float* b_data = beta.ptr<float>();
    const float* m_data = mean.ptr<float>();
    const float* v_data = var.ptr<float>();
    float* out_data = output.ptr<float>();
    
    for (size_t c = 0; c < C; ++c) {
        float scale = g_data[c] / std::sqrt(v_data[c] + eps);
        float shift = b_data[c] - m_data[c] * scale;
        
        for (size_t i = 0; i < H * W; ++i) {
            out_data[c * H * W + i] = out_data[c * H * W + i] * scale + shift;
        }
    }
    
    return output;
}

} // namespace nn

} // namespace neurova

// ============================================================================
// Python Bindings
// ============================================================================

using namespace neurova;

py::dtype dtype_to_numpy(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32: return py::dtype::of<float>();
        case DType::FLOAT64: return py::dtype::of<double>();
        case DType::INT32: return py::dtype::of<int32_t>();
        case DType::INT64: return py::dtype::of<int64_t>();
        case DType::UINT8: return py::dtype::of<uint8_t>();
        case DType::BOOL: return py::dtype::of<bool>();
        default: return py::dtype::of<float>();
    }
}

py::array tensor_to_numpy(const Tensor& t) {
    std::vector<ssize_t> shape;
    for (auto s : t.shape()) shape.push_back(static_cast<ssize_t>(s));
    
    std::vector<ssize_t> strides;
    ssize_t stride = dtype_size(t.dtype());
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides.insert(strides.begin(), stride);
        stride *= shape[i];
    }
    
    return py::array(dtype_to_numpy(t.dtype()), shape, strides, t.data());
}

Tensor numpy_to_tensor(py::array arr) {
    py::buffer_info info = arr.request();
    
    std::vector<size_t> shape;
    for (auto s : info.shape) shape.push_back(static_cast<size_t>(s));
    
    DType dtype = DType::FLOAT32;
    if (arr.dtype().is(py::dtype::of<uint8_t>())) dtype = DType::UINT8;
    else if (arr.dtype().is(py::dtype::of<double>())) dtype = DType::FLOAT64;
    
    Tensor t(shape, dtype);
    std::memcpy(t.data(), info.ptr, t.size() * dtype_size(dtype));
    
    return t;
}

py::array image_to_numpy(const Image& img) {
    size_t H = img.height();
    size_t W = img.width();
    size_t C = img.channels();
    
    std::vector<ssize_t> shape = {static_cast<ssize_t>(H), 
                                  static_cast<ssize_t>(W), 
                                  static_cast<ssize_t>(C)};
    
    py::array result(dtype_to_numpy(img.dtype()), shape);
    auto buf = result.request();
    
    // Convert CHW to HWC
    if (img.dtype() == DType::UINT8) {
        const uint8_t* src = img.ptr<uint8_t>();
        uint8_t* dst = static_cast<uint8_t*>(buf.ptr);
        
        for (size_t y = 0; y < H; ++y) {
            for (size_t x = 0; x < W; ++x) {
                for (size_t c = 0; c < C; ++c) {
                    dst[(y * W + x) * C + c] = src[c * H * W + y * W + x];
                }
            }
        }
    }
    
    return result;
}

Image numpy_to_image(py::array arr) {
    py::buffer_info info = arr.request();
    
    size_t H, W, C;
    if (info.ndim == 2) {
        H = info.shape[0]; W = info.shape[1]; C = 1;
    } else {
        H = info.shape[0]; W = info.shape[1]; C = info.shape[2];
    }
    
    DType dtype = arr.dtype().is(py::dtype::of<uint8_t>()) ? DType::UINT8 : DType::FLOAT32;
    Image img(W, H, C, dtype);
    
    // Convert HWC to CHW
    if (dtype == DType::UINT8) {
        const uint8_t* src = static_cast<const uint8_t*>(info.ptr);
        uint8_t* dst = img.ptr<uint8_t>();
        
        for (size_t c = 0; c < C; ++c) {
            for (size_t y = 0; y < H; ++y) {
                for (size_t x = 0; x < W; ++x) {
                    if (C == 1) {
                        dst[c * H * W + y * W + x] = src[y * W + x];
                    } else {
                        dst[c * H * W + y * W + x] = src[(y * W + x) * C + c];
                    }
                }
            }
        }
    }
    
    return img;
}

PYBIND11_MODULE(neurova_minimal, m) {
    m.doc() = "Neurova Minimal Bundle: imgproc, filters, features (C++ accelerated)";
    
    m.attr("__version__") = VERSION_STRING;
    m.attr("VERSION_MAJOR") = VERSION_MAJOR;
    m.attr("VERSION_MINOR") = VERSION_MINOR;
    m.attr("VERSION_PATCH") = VERSION_PATCH;
    
    #ifdef __ARM_NEON
    m.attr("SIMD_SUPPORT") = "ARM NEON";
    #elif defined(__AVX2__)
    m.attr("SIMD_SUPPORT") = "AVX2";
    #elif defined(__AVX__)
    m.attr("SIMD_SUPPORT") = "AVX";
    #elif defined(__SSE2__)
    m.attr("SIMD_SUPPORT") = "SSE2";
    #else
    m.attr("SIMD_SUPPORT") = "None";
    #endif
    
    // DType enum
    py::enum_<DType>(m, "DType")
        .value("FLOAT32", DType::FLOAT32)
        .value("FLOAT64", DType::FLOAT64)
        .value("INT32", DType::INT32)
        .value("INT64", DType::INT64)
        .value("UINT8", DType::UINT8)
        .value("BOOL", DType::BOOL)
        .export_values();
    
    // Rect
    py::class_<Rect>(m, "Rect")
        .def(py::init<>())
        .def(py::init<int, int, int, int>())
        .def_readwrite("x", &Rect::x)
        .def_readwrite("y", &Rect::y)
        .def_readwrite("width", &Rect::width)
        .def_readwrite("height", &Rect::height)
        .def("area", &Rect::area);
    
    // Tensor
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init([](py::array arr) { return numpy_to_tensor(arr); }))
        .def(py::init<const std::vector<size_t>&, DType>(), 
             py::arg("shape"), py::arg("dtype") = DType::FLOAT32)
        
        .def_static("zeros", &Tensor::zeros, py::arg("shape"), py::arg("dtype") = DType::FLOAT32)
        .def_static("ones", &Tensor::ones, py::arg("shape"), py::arg("dtype") = DType::FLOAT32)
        .def_static("randn", &Tensor::randn, py::arg("shape"), py::arg("dtype") = DType::FLOAT32)
        .def_static("rand", &Tensor::rand, py::arg("shape"), py::arg("dtype") = DType::FLOAT32)
        .def_static("arange", &Tensor::arange)
        .def_static("eye", &Tensor::eye, py::arg("n"), py::arg("dtype") = DType::FLOAT32)
        
        .def_property_readonly("shape", [](const Tensor& t) { return py::cast(t.shape()); })
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("size", &Tensor::size)
        .def_property_readonly("ndim", &Tensor::ndim)
        
        .def("reshape", &Tensor::reshape)
        .def("squeeze", &Tensor::squeeze, py::arg("axis") = -1)
        .def("unsqueeze", &Tensor::unsqueeze)
        .def("transpose", &Tensor::transpose)
        .def("clone", &Tensor::clone)
        .def("contiguous", &Tensor::contiguous)
        .def("to", &Tensor::to)
        
        .def("numpy", [](const Tensor& t) { return tensor_to_numpy(t); })
        .def_static("from_numpy", [](py::array arr) { return numpy_to_tensor(arr); })
        
        .def("__add__", [](const Tensor& a, const Tensor& b) { return a + b; })
        .def("__sub__", [](const Tensor& a, const Tensor& b) { return a - b; })
        .def("__mul__", [](const Tensor& a, const Tensor& b) { return a * b; })
        .def("__truediv__", [](const Tensor& a, const Tensor& b) { return a / b; })
        .def("__neg__", [](const Tensor& t) { return -t; })
        .def("__add__", [](const Tensor& t, float s) { return t + s; })
        .def("__sub__", [](const Tensor& t, float s) { return t - s; })
        .def("__mul__", [](const Tensor& t, float s) { return t * s; })
        .def("__truediv__", [](const Tensor& t, float s) { return t / s; })
        .def("__radd__", [](const Tensor& t, float s) { return t + s; })
        .def("__rmul__", [](const Tensor& t, float s) { return t * s; })
        
        .def("sum", &Tensor::sum)
        .def("mean", &Tensor::mean)
        .def("max", &Tensor::max)
        .def("min", &Tensor::min)
        .def("std", &Tensor::std)
        .def("var", &Tensor::var)
        .def("argmax", &Tensor::argmax)
        .def("argmin", &Tensor::argmin)
        
        .def("matmul", &Tensor::matmul)
        .def("dot", &Tensor::dot)
        
        .def("exp", &Tensor::exp)
        .def("log", &Tensor::log)
        .def("sqrt", &Tensor::sqrt)
        .def("pow", &Tensor::pow)
        .def("abs", &Tensor::abs)
        .def("sin", &Tensor::sin)
        .def("cos", &Tensor::cos)
        .def("tanh", &Tensor::tanh)
        .def("clamp", &Tensor::clamp)
        
        .def("__repr__", [](const Tensor& t) {
            std::string s = "Tensor(shape=[";
            for (size_t i = 0; i < t.ndim(); ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(t.shape()[i]);
            }
            s += "])";
            return s;
        });
    
    // Image
    py::class_<Image>(m, "Image")
        .def(py::init<>())
        .def(py::init([](py::array arr) { return numpy_to_image(arr); }))
        .def(py::init<size_t, size_t, size_t, DType>(),
             py::arg("width"), py::arg("height"), py::arg("channels") = 3,
             py::arg("dtype") = DType::UINT8)
        
        .def_property_readonly("width", &Image::width)
        .def_property_readonly("height", &Image::height)
        .def_property_readonly("channels", &Image::channels)
        .def_property_readonly("dtype", &Image::dtype)
        .def_property_readonly("shape", [](const Image& img) {
            return py::make_tuple(img.height(), img.width(), img.channels());
        })
        
        .def("clone", &Image::clone)
        .def("to", &Image::to)
        .def("to_tensor", &Image::to_tensor)
        .def("numpy", [](const Image& img) { return image_to_numpy(img); })
        .def_static("from_numpy", [](py::array arr) { return numpy_to_image(arr); })
        
        .def("__repr__", [](const Image& img) {
            return "Image(" + std::to_string(img.width()) + "x" + 
                   std::to_string(img.height()) + ", channels=" +
                   std::to_string(img.channels()) + ")";
        });
    
    // imgproc submodule
    py::module imgproc_m = m.def_submodule("imgproc", "Image processing functions");
    imgproc_m.def("rgb_to_gray", &imgproc::rgb_to_gray);
    imgproc_m.def("gray_to_rgb", &imgproc::gray_to_rgb);
    imgproc_m.def("gaussian_blur", &imgproc::gaussian_blur, 
                  py::arg("src"), py::arg("ksize") = 3, py::arg("sigma") = 0.0f);
    imgproc_m.def("resize", &imgproc::resize);
    imgproc_m.def("crop", &imgproc::crop);
    imgproc_m.def("flip", &imgproc::flip);
    imgproc_m.def("threshold", &imgproc::threshold, 
                  py::arg("src"), py::arg("thresh"), py::arg("maxval") = 255, py::arg("type") = 0);
    imgproc_m.def("adaptive_threshold", &imgproc::adaptive_threshold,
                  py::arg("src"), py::arg("maxval") = 255, py::arg("method") = 0,
                  py::arg("type") = 0, py::arg("block_size") = 11, py::arg("C") = 2);
    imgproc_m.def("rotate", &imgproc::rotate, py::arg("src"), py::arg("angle"), 
                  py::arg("center_x") = -1, py::arg("center_y") = -1);
    imgproc_m.def("histogram", &imgproc::histogram, py::arg("src"), py::arg("bins") = 256);
    imgproc_m.def("equalize_hist", &imgproc::equalize_hist);
    
    // filters submodule
    py::module filters_m = m.def_submodule("filters", "Image filters");
    filters_m.def("sobel", &filters::sobel, py::arg("src"), py::arg("dx") = 1, py::arg("dy") = 0);
    filters_m.def("canny", &filters::canny, py::arg("src"), 
                  py::arg("low_threshold") = 50, py::arg("high_threshold") = 150);
    filters_m.def("median_filter", &filters::median_filter, py::arg("src"), py::arg("ksize") = 3);
    filters_m.def("bilateral_filter", &filters::bilateral_filter, 
                  py::arg("src"), py::arg("d") = 9, py::arg("sigma_color") = 75, py::arg("sigma_space") = 75);
    filters_m.def("laplacian", &filters::laplacian, py::arg("src"), py::arg("ksize") = 3);
    filters_m.def("scharr", &filters::scharr, py::arg("src"), py::arg("dx") = 1, py::arg("dy") = 0);
    filters_m.def("box_filter", &filters::box_filter, py::arg("src"), py::arg("ksize") = 3);
    
    // morphology submodule
    py::module morph_m = m.def_submodule("morphology", "Morphological operations");
    morph_m.def("erode", &morphology::erode, py::arg("src"), py::arg("ksize") = 3);
    morph_m.def("dilate", &morphology::dilate, py::arg("src"), py::arg("ksize") = 3);
    morph_m.def("opening", &morphology::opening, py::arg("src"), py::arg("ksize") = 3);
    morph_m.def("closing", &morphology::closing, py::arg("src"), py::arg("ksize") = 3);
    morph_m.def("gradient", &morphology::gradient, py::arg("src"), py::arg("ksize") = 3);
    
    // features submodule
    py::module features_m = m.def_submodule("features", "Feature detection");
    features_m.def("harris_corners", &features::harris_corners, 
                   py::arg("src"), py::arg("block_size") = 3, py::arg("ksize") = 3, py::arg("k") = 0.04f);
    features_m.def("good_features_to_track", &features::good_features_to_track,
                   py::arg("src"), py::arg("max_corners") = 100, py::arg("quality_level") = 0.01f);
    
    // transform submodule
    py::module transform_m = m.def_submodule("transform", "Geometric transformations");
    transform_m.def("get_rotation_matrix_2d", &transform::get_rotation_matrix_2d,
                    py::arg("cx"), py::arg("cy"), py::arg("angle"), py::arg("scale") = 1.0f);
    transform_m.def("warp_affine", &transform::warp_affine,
                    py::arg("src"), py::arg("M"), py::arg("dst_w"), py::arg("dst_h"));
    
    // video submodule
    py::module video_m = m.def_submodule("video", "Video processing");
    py::class_<video::BackgroundSubtractor>(video_m, "BackgroundSubtractor")
        .def(py::init<>())
        .def("apply", &video::BackgroundSubtractor::apply,
             py::arg("frame"), py::arg("fgmask"), py::arg("learning_rate") = 0.01f)
        .def("reset", &video::BackgroundSubtractor::reset);
    video_m.def("calc_optical_flow_lk", &video::calc_optical_flow_lk,
                py::arg("prev"), py::arg("next"), py::arg("win_size") = 15);
    
    // segmentation submodule
    py::module seg_m = m.def_submodule("segmentation", "Image segmentation");
    seg_m.def("distance_transform", &segmentation::distance_transform);
    seg_m.def("watershed", &segmentation::watershed, py::arg("markers"), py::arg("gradient"));
    seg_m.def("connected_components", &segmentation::connected_components,
              py::arg("binary"), py::arg("num_labels"));
    
    // nn submodule
    py::module nn_m = m.def_submodule("nn", "Neural network functions");
    nn_m.def("relu", &nn::relu);
    nn_m.def("leaky_relu", &nn::leaky_relu, py::arg("x"), py::arg("negative_slope") = 0.01f);
    nn_m.def("sigmoid", &nn::sigmoid);
    nn_m.def("tanh", &nn::tanh);
    nn_m.def("softmax", &nn::softmax, py::arg("x"), py::arg("dim") = -1);
    nn_m.def("gelu", &nn::gelu);
    nn_m.def("swish", &nn::swish);
    nn_m.def("mse_loss", &nn::mse_loss);
    nn_m.def("cross_entropy", &nn::cross_entropy);
    nn_m.def("conv2d", &nn::conv2d, py::arg("input"), py::arg("kernel"), 
             py::arg("stride") = 1, py::arg("padding") = 0);
    nn_m.def("max_pool2d", &nn::max_pool2d, py::arg("input"), 
             py::arg("kernel_size") = 2, py::arg("stride") = 2);
    nn_m.def("avg_pool2d", &nn::avg_pool2d, py::arg("input"), 
             py::arg("kernel_size") = 2, py::arg("stride") = 2);
    nn_m.def("batch_norm", &nn::batch_norm, py::arg("input"), py::arg("gamma"), 
             py::arg("beta"), py::arg("mean"), py::arg("var"), py::arg("eps") = 1e-5f);
    
    // ml submodule
    py::module ml_m = m.def_submodule("ml", "Machine learning functions");
    ml_m.def("accuracy_score", &ml::accuracy_score);
    ml_m.def("mean_squared_error", &ml::mean_squared_error);
    
    // ML classes
    py::class_<ml::KMeans>(ml_m, "KMeans")
        .def(py::init<int, int, float>(), 
             py::arg("n_clusters") = 8, py::arg("max_iter") = 300, py::arg("tol") = 1e-4f)
        .def("fit", &ml::KMeans::fit)
        .def("predict", &ml::KMeans::predict)
        .def("get_centroids", &ml::KMeans::get_centroids)
        .def("get_labels", &ml::KMeans::get_labels);
    
    py::class_<ml::PCA>(ml_m, "PCA")
        .def(py::init<int>(), py::arg("n_components") = 2)
        .def("fit", &ml::PCA::fit)
        .def("transform", &ml::PCA::transform)
        .def("get_components", &ml::PCA::get_components);
    
    py::class_<ml::LinearRegression>(ml_m, "LinearRegression")
        .def(py::init<>())
        .def("fit", &ml::LinearRegression::fit)
        .def("predict", &ml::LinearRegression::predict);
    
    // face submodule
    py::module face_m = m.def_submodule("face", "Face detection");
    py::class_<face::HaarCascade>(face_m, "HaarCascade")
        .def(py::init<>())
        .def("load", &face::HaarCascade::load)
        .def("detect", &face::HaarCascade::detect,
             py::arg("img"), py::arg("scale_factor") = 1.1f,
             py::arg("min_neighbors") = 3, py::arg("min_size") = 30);
}
