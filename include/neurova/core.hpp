// Copyright (c) 2026 @squid consultancy group (scg)
// all rights reserved.
// licensed under the apache license 2.0.

/**
 * neurova/core.hpp - Core data structures and operations
 * 
 * This header defines the fundamental types used throughout Neurova:
 * - Image: N-dimensional array for image data
 * - Matrix: 2D matrix with linear algebra operations
 * - Tensor: N-dimensional tensor for neural networks
 */

#ifndef NEUROVA_CORE_HPP
#define NEUROVA_CORE_HPP

#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <functional>

namespace neurova {

// ============================================================================
// Data Types
// ============================================================================

enum class DType {
    UINT8,
    INT8,
    UINT16,
    INT16,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    BOOL
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::UINT8:
        case DType::INT8:
        case DType::BOOL:
            return 1;
        case DType::UINT16:
        case DType::INT16:
            return 2;
        case DType::INT32:
        case DType::FLOAT32:
            return 4;
        case DType::INT64:
        case DType::FLOAT64:
            return 8;
        default:
            return 1;
    }
}

// ============================================================================
// Shape class
// ============================================================================

class Shape {
public:
    std::vector<size_t> dims;
    
    Shape() = default;
    Shape(std::initializer_list<size_t> d) : dims(d) {}
    Shape(const std::vector<size_t>& d) : dims(d) {}
    
    size_t ndim() const { return dims.size(); }
    size_t operator[](size_t i) const { return dims[i]; }
    size_t& operator[](size_t i) { return dims[i]; }
    
    size_t total() const {
        if (dims.empty()) return 0;
        size_t t = 1;
        for (auto d : dims) t *= d;
        return t;
    }
    
    bool operator==(const Shape& other) const {
        return dims == other.dims;
    }
    
    bool operator!=(const Shape& other) const {
        return dims != other.dims;
    }
    
    void push_back(size_t d) { dims.push_back(d); }
    auto begin() const { return dims.begin(); }
    auto end() const { return dims.end(); }
    auto begin() { return dims.begin(); }
    auto end() { return dims.end(); }
    bool empty() const { return dims.empty(); }
    size_t size() const { return dims.size(); }
};

// ============================================================================
// Tensor class - N-dimensional array
// ============================================================================

class Tensor {
public:
    // constructors
    Tensor();
    Tensor(const Shape& shape, DType dtype = DType::FLOAT32);
    Tensor(const std::vector<size_t>& shape, DType dtype = DType::FLOAT32);
    Tensor(size_t rows, size_t cols, DType dtype = DType::FLOAT32);
    Tensor(size_t height, size_t width, size_t channels, DType dtype = DType::UINT8);
    
    // copy/move
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    ~Tensor() = default;
    
    // factory methods
    static Tensor zeros(const Shape& shape, DType dtype = DType::FLOAT32);
    static Tensor ones(const Shape& shape, DType dtype = DType::FLOAT32);
    static Tensor full(const Shape& shape, double value, DType dtype = DType::FLOAT32);
    static Tensor eye(size_t n, DType dtype = DType::FLOAT32);
    static Tensor randn(const Shape& shape, DType dtype = DType::FLOAT32);
    static Tensor rand(const Shape& shape, DType dtype = DType::FLOAT32);
    static Tensor arange(double start, double stop, double step = 1.0, DType dtype = DType::FLOAT32);
    static Tensor linspace(double start, double stop, size_t num, DType dtype = DType::FLOAT32);
    
    // properties
    const Shape& shape() const { return shape_; }
    size_t ndim() const { return shape_.ndim(); }
    size_t size() const { return size_; }
    DType dtype() const { return dtype_; }
    size_t itemsize() const { return dtype_size(dtype_); }
    size_t nbytes() const { return size_ * itemsize(); }
    bool empty() const { return size_ == 0; }
    
    // data access
    void* data() { return data_.get(); }
    const void* data() const { return data_.get(); }
    
    template<typename T>
    T* ptr() { return static_cast<T*>(data_.get()); }
    
    template<typename T>
    const T* ptr() const { return static_cast<const T*>(data_.get()); }
    
    // element access
    template<typename T>
    T& at(size_t i) { return ptr<T>()[i]; }
    
    template<typename T>
    const T& at(size_t i) const { return ptr<T>()[i]; }
    
    template<typename T>
    T& at(size_t i, size_t j) {
        return ptr<T>()[i * shape_[1] + j];
    }
    
    template<typename T>
    const T& at(size_t i, size_t j) const {
        return ptr<T>()[i * shape_[1] + j];
    }
    
    template<typename T>
    T& at(size_t i, size_t j, size_t k) {
        return ptr<T>()[(i * shape_[1] + j) * shape_[2] + k];
    }
    
    template<typename T>
    const T& at(size_t i, size_t j, size_t k) const {
        return ptr<T>()[(i * shape_[1] + j) * shape_[2] + k];
    }
    
    // reshape
    Tensor reshape(const Shape& new_shape) const;
    Tensor flatten() const;
    Tensor transpose() const;
    Tensor transpose(const std::vector<size_t>& axes) const;
    
    // slicing
    Tensor slice(size_t start, size_t end, size_t axis = 0) const;
    
    // type conversion
    Tensor astype(DType new_dtype) const;
    Tensor to(DType new_dtype) const { return astype(new_dtype); }
    Tensor to_float32() const { return astype(DType::FLOAT32); }
    Tensor to_float64() const { return astype(DType::FLOAT64); }
    Tensor to_uint8() const { return astype(DType::UINT8); }
    
    // arithmetic operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    Tensor operator+(double scalar) const;
    Tensor operator-(double scalar) const;
    Tensor operator*(double scalar) const;
    Tensor operator/(double scalar) const;
    
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    
    // matrix operations
    Tensor matmul(const Tensor& other) const;
    Tensor dot(const Tensor& other) const { return matmul(other); }
    
    // reduction operations
    double sum() const;
    double mean() const;
    double min() const;
    double max() const;
    double std() const;
    double var() const;
    
    Tensor sum(int axis, bool keepdims = false) const;
    Tensor mean(int axis, bool keepdims = false) const;
    Tensor min(int axis, bool keepdims = false) const;
    Tensor max(int axis, bool keepdims = false) const;
    
    size_t argmin() const;
    size_t argmax() const;
    Tensor argmin(int axis) const;
    Tensor argmax(int axis) const;
    
    // element-wise math
    Tensor abs() const;
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor pow(double exponent) const;
    Tensor clip(double min_val, double max_val) const;
    
    // comparison
    Tensor operator>(double val) const;
    Tensor operator<(double val) const;
    Tensor operator>=(double val) const;
    Tensor operator<=(double val) const;
    Tensor operator==(double val) const;
    
    // copy
    Tensor clone() const;
    Tensor contiguous() const;

private:
    std::shared_ptr<void> data_;
    Shape shape_;
    size_t size_;
    DType dtype_;
    std::vector<size_t> strides_;
    
    void allocate();
    void compute_strides();
};

// ============================================================================
// Image class - Specialized for image data (H, W, C)
// ============================================================================

class Image {
public:
    Image();
    Image(size_t height, size_t width, size_t channels = 3, DType dtype = DType::UINT8);
    Image(const Tensor& tensor);
    
    // factory methods
    static Image zeros(size_t height, size_t width, size_t channels = 3);
    static Image ones(size_t height, size_t width, size_t channels = 3);
    
    // properties
    size_t height() const { return data_.shape()[0]; }
    size_t width() const { return data_.shape()[1]; }
    size_t channels() const { return data_.ndim() > 2 ? data_.shape()[2] : 1; }
    Shape shape() const { return data_.shape(); }
    DType dtype() const { return data_.dtype(); }
    bool empty() const { return data_.empty(); }
    size_t size() const { return data_.size(); }
    
    // data access
    void* data() { return data_.data(); }
    const void* data() const { return data_.data(); }
    Tensor& tensor() { return data_; }
    const Tensor& tensor() const { return data_; }
    
    template<typename T>
    T* ptr() { return data_.ptr<T>(); }
    
    template<typename T>
    const T* ptr() const { return data_.ptr<T>(); }
    
    template<typename T>
    T& at(size_t y, size_t x) { return data_.at<T>(y, x); }
    
    template<typename T>
    const T& at(size_t y, size_t x) const { return data_.at<T>(y, x); }
    
    template<typename T>
    T& at(size_t y, size_t x, size_t c) { return data_.at<T>(y, x, c); }
    
    template<typename T>
    const T& at(size_t y, size_t x, size_t c) const { return data_.at<T>(y, x, c); }
    
    // conversion
    Image to_grayscale() const;
    Image to_rgb() const;
    Image to_float() const;
    Image to_uint8() const;
    Image to(DType new_dtype) const;
    Tensor to_tensor() const { return data_; }
    
    // clone
    Image clone() const;

private:
    Tensor data_;
};

// ============================================================================
// Utility functions
// ============================================================================

// element-wise operations
Tensor add(const Tensor& a, const Tensor& b);
Tensor subtract(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor divide(const Tensor& a, const Tensor& b);

// matrix operations
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor transpose(const Tensor& a);
Tensor inv(const Tensor& a);
Tensor det(const Tensor& a);
std::pair<Tensor, Tensor> eig(const Tensor& a);
std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& a);

// math functions
Tensor abs(const Tensor& a);
Tensor sqrt(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor pow(const Tensor& a, double exp);
Tensor sin(const Tensor& a);
Tensor cos(const Tensor& a);
Tensor tan(const Tensor& a);

// reduction
double sum(const Tensor& a);
double mean(const Tensor& a);
double min(const Tensor& a);
double max(const Tensor& a);
double std(const Tensor& a);
double var(const Tensor& a);

// concatenation
Tensor concatenate(const std::vector<Tensor>& tensors, int axis = 0);
Tensor stack(const std::vector<Tensor>& tensors, int axis = 0);
Tensor vstack(const std::vector<Tensor>& tensors);
Tensor hstack(const std::vector<Tensor>& tensors);

// splitting
std::vector<Tensor> split(const Tensor& a, size_t num_splits, int axis = 0);

} // namespace neurova

#endif // NEUROVA_CORE_HPP
