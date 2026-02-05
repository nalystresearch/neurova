// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file tensor.hpp
 * @brief Tensor class with automatic differentiation support
 * 
 * Neurova implementation of autograd-enabled tensors for neural networks.
 * Supports reverse-mode automatic differentiation (backpropagation).
 */

#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <initializer_list>

namespace neurova {
namespace nn {

// Forward declarations
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

/**
 * @brief Tensor with automatic differentiation support
 * 
 * Tracks computational graph for automatic backpropagation.
 */
class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<float> data;
    std::vector<size_t> shape;
    bool requires_grad = false;
    std::vector<float> grad;
    
private:
    std::function<void()> backward_fn_;
    std::vector<TensorPtr> prev_;
    std::string op_;
    
public:
    // Constructors
    Tensor() = default;
    
    explicit Tensor(float value, bool requires_grad = false)
        : data({value}), shape({1}), requires_grad(requires_grad) {
        if (requires_grad) {
            grad.resize(1, 0.0f);
        }
    }
    
    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad = false)
        : data(data), shape(shape), requires_grad(requires_grad) {
        if (requires_grad) {
            grad.resize(data.size(), 0.0f);
        }
    }
    
    Tensor(std::initializer_list<float> values, bool requires_grad = false)
        : data(values), shape({values.size()}), requires_grad(requires_grad) {
        if (requires_grad) {
            grad.resize(data.size(), 0.0f);
        }
    }
    
    // Factory methods
    static TensorPtr create(const std::vector<float>& data, const std::vector<size_t>& shape, 
                            bool requires_grad = false) {
        auto t = std::make_shared<Tensor>(data, shape, requires_grad);
        return t;
    }
    
    static TensorPtr zeros(const std::vector<size_t>& shape, bool requires_grad = false) {
        size_t total = 1;
        for (auto s : shape) total *= s;
        return std::make_shared<Tensor>(std::vector<float>(total, 0.0f), shape, requires_grad);
    }
    
    static TensorPtr ones(const std::vector<size_t>& shape, bool requires_grad = false) {
        size_t total = 1;
        for (auto s : shape) total *= s;
        return std::make_shared<Tensor>(std::vector<float>(total, 1.0f), shape, requires_grad);
    }
    
    static TensorPtr randn(const std::vector<size_t>& shape, bool requires_grad = false) {
        size_t total = 1;
        for (auto s : shape) total *= s;
        std::vector<float> data(total);
        // Box-Muller transform for normal distribution
        for (size_t i = 0; i < total; i += 2) {
            float u1 = static_cast<float>(rand()) / RAND_MAX;
            float u2 = static_cast<float>(rand()) / RAND_MAX;
            u1 = std::max(u1, 1e-7f);
            float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
            float z1 = std::sqrt(-2.0f * std::log(u1)) * std::sin(2.0f * 3.14159265f * u2);
            data[i] = z0;
            if (i + 1 < total) data[i + 1] = z1;
        }
        return std::make_shared<Tensor>(data, shape, requires_grad);
    }
    
    static TensorPtr uniform(const std::vector<size_t>& shape, float low, float high, 
                             bool requires_grad = false) {
        size_t total = 1;
        for (auto s : shape) total *= s;
        std::vector<float> data(total);
        for (size_t i = 0; i < total; ++i) {
            data[i] = low + (high - low) * static_cast<float>(rand()) / RAND_MAX;
        }
        return std::make_shared<Tensor>(data, shape, requires_grad);
    }
    
    // Properties
    size_t ndim() const { return shape.size(); }
    
    size_t size() const {
        size_t total = 1;
        for (auto s : shape) total *= s;
        return total;
    }
    
    size_t numel() const { return size(); }
    
    // Element access
    float& operator[](size_t idx) { return data[idx]; }
    const float& operator[](size_t idx) const { return data[idx]; }
    
    float& at(const std::vector<size_t>& indices) {
        return data[flatIndex(indices)];
    }
    
    const float& at(const std::vector<size_t>& indices) const {
        return data[flatIndex(indices)];
    }
    
    // Reshape
    TensorPtr reshape(const std::vector<size_t>& new_shape) const {
        size_t new_size = 1;
        for (auto s : new_shape) new_size *= s;
        if (new_size != size()) {
            throw std::runtime_error("Cannot reshape tensor to incompatible size");
        }
        return std::make_shared<Tensor>(data, new_shape, requires_grad);
    }
    
    TensorPtr view(const std::vector<size_t>& new_shape) const {
        return reshape(new_shape);
    }
    
    TensorPtr flatten() const {
        return reshape({size()});
    }
    
    TensorPtr transpose() const {
        if (shape.size() != 2) {
            throw std::runtime_error("Transpose only supported for 2D tensors");
        }
        size_t rows = shape[0], cols = shape[1];
        std::vector<float> result(data.size());
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[j * rows + i] = data[i * cols + j];
            }
        }
        return std::make_shared<Tensor>(result, {cols, rows}, requires_grad);
    }
    
    TensorPtr T() const { return transpose(); }
    
    // Arithmetic operations with autograd
    TensorPtr add(const TensorPtr& other) const {
        auto out = std::make_shared<Tensor>();
        out->shape = broadcastShape(shape, other->shape);
        out->data.resize(out->size());
        out->requires_grad = requires_grad || other->requires_grad;
        if (out->requires_grad) out->grad.resize(out->size(), 0.0f);
        
        // Broadcast add
        for (size_t i = 0; i < out->size(); ++i) {
            out->data[i] = getBroadcast(i, out->shape) + other->getBroadcast(i, out->shape);
        }
        
        if (out->requires_grad) {
            auto self_ptr = std::const_pointer_cast<Tensor>(shared_from_this());
            out->prev_ = {self_ptr, other};
            out->op_ = "+";
            auto out_weak = std::weak_ptr<Tensor>(out);
            auto self_weak = std::weak_ptr<Tensor>(self_ptr);
            auto other_weak = std::weak_ptr<Tensor>(other);
            
            out->backward_fn_ = [out_weak, self_weak, other_weak]() {
                auto out = out_weak.lock();
                auto self = self_weak.lock();
                auto other = other_weak.lock();
                if (!out || !self || !other) return;
                
                if (self->requires_grad) {
                    for (size_t i = 0; i < out->size(); ++i) {
                        size_t self_idx = self->broadcastIndex(i, out->shape);
                        self->grad[self_idx] += out->grad[i];
                    }
                }
                if (other->requires_grad) {
                    for (size_t i = 0; i < out->size(); ++i) {
                        size_t other_idx = other->broadcastIndex(i, out->shape);
                        other->grad[other_idx] += out->grad[i];
                    }
                }
            };
        }
        return out;
    }
    
    TensorPtr sub(const TensorPtr& other) const {
        auto neg = other->neg();
        return add(neg);
    }
    
    TensorPtr mul(const TensorPtr& other) const {
        auto out = std::make_shared<Tensor>();
        out->shape = broadcastShape(shape, other->shape);
        out->data.resize(out->size());
        out->requires_grad = requires_grad || other->requires_grad;
        if (out->requires_grad) out->grad.resize(out->size(), 0.0f);
        
        for (size_t i = 0; i < out->size(); ++i) {
            out->data[i] = getBroadcast(i, out->shape) * other->getBroadcast(i, out->shape);
        }
        
        if (out->requires_grad) {
            auto self_ptr = std::const_pointer_cast<Tensor>(shared_from_this());
            out->prev_ = {self_ptr, other};
            out->op_ = "*";
            auto out_weak = std::weak_ptr<Tensor>(out);
            auto self_weak = std::weak_ptr<Tensor>(self_ptr);
            auto other_weak = std::weak_ptr<Tensor>(other);
            
            out->backward_fn_ = [out_weak, self_weak, other_weak]() {
                auto out = out_weak.lock();
                auto self = self_weak.lock();
                auto other = other_weak.lock();
                if (!out || !self || !other) return;
                
                if (self->requires_grad) {
                    for (size_t i = 0; i < out->size(); ++i) {
                        size_t self_idx = self->broadcastIndex(i, out->shape);
                        self->grad[self_idx] += out->grad[i] * other->getBroadcast(i, out->shape);
                    }
                }
                if (other->requires_grad) {
                    for (size_t i = 0; i < out->size(); ++i) {
                        size_t other_idx = other->broadcastIndex(i, out->shape);
                        other->grad[other_idx] += out->grad[i] * self->getBroadcast(i, out->shape);
                    }
                }
            };
        }
        return out;
    }
    
    TensorPtr div(const TensorPtr& other) const {
        auto inv = other->pow(-1.0f);
        return mul(inv);
    }
    
    TensorPtr neg() const {
        auto out = std::make_shared<Tensor>(data, shape, requires_grad);
        for (auto& v : out->data) v = -v;
        
        if (requires_grad) {
            out->grad.resize(out->size(), 0.0f);
            auto self_ptr = std::const_pointer_cast<Tensor>(shared_from_this());
            out->prev_ = {self_ptr};
            out->op_ = "neg";
            auto out_weak = std::weak_ptr<Tensor>(out);
            auto self_weak = std::weak_ptr<Tensor>(self_ptr);
            
            out->backward_fn_ = [out_weak, self_weak]() {
                auto out = out_weak.lock();
                auto self = self_weak.lock();
                if (!out || !self) return;
                for (size_t i = 0; i < self->size(); ++i) {
                    self->grad[i] += -out->grad[i];
                }
            };
        }
        return out;
    }
    
    TensorPtr pow(float exp) const {
        auto out = std::make_shared<Tensor>(data, shape, requires_grad);
        for (auto& v : out->data) v = std::pow(v, exp);
        
        if (requires_grad) {
            out->grad.resize(out->size(), 0.0f);
            auto self_ptr = std::const_pointer_cast<Tensor>(shared_from_this());
            out->prev_ = {self_ptr};
            out->op_ = "pow";
            auto out_weak = std::weak_ptr<Tensor>(out);
            auto self_weak = std::weak_ptr<Tensor>(self_ptr);
            
            out->backward_fn_ = [out_weak, self_weak, exp]() {
                auto out = out_weak.lock();
                auto self = self_weak.lock();
                if (!out || !self) return;
                for (size_t i = 0; i < self->size(); ++i) {
                    self->grad[i] += exp * std::pow(self->data[i], exp - 1) * out->grad[i];
                }
            };
        }
        return out;
    }
    
    // Matrix multiplication
    TensorPtr matmul(const TensorPtr& other) const {
        if (shape.size() != 2 || other->shape.size() != 2) {
            throw std::runtime_error("matmul requires 2D tensors");
        }
        if (shape[1] != other->shape[0]) {
            throw std::runtime_error("matmul dimension mismatch");
        }
        
        size_t M = shape[0], K = shape[1], N = other->shape[1];
        std::vector<float> result(M * N, 0.0f);
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                for (size_t k = 0; k < K; ++k) {
                    result[i * N + j] += data[i * K + k] * other->data[k * N + j];
                }
            }
        }
        
        auto out = std::make_shared<Tensor>(result, {M, N}, requires_grad || other->requires_grad);
        
        if (out->requires_grad) {
            auto self_ptr = std::const_pointer_cast<Tensor>(shared_from_this());
            out->prev_ = {self_ptr, other};
            out->op_ = "@";
            auto out_weak = std::weak_ptr<Tensor>(out);
            auto self_weak = std::weak_ptr<Tensor>(self_ptr);
            auto other_weak = std::weak_ptr<Tensor>(other);
            
            out->backward_fn_ = [out_weak, self_weak, other_weak, M, K, N]() {
                auto out = out_weak.lock();
                auto self = self_weak.lock();
                auto other = other_weak.lock();
                if (!out || !self || !other) return;
                
                if (self->requires_grad) {
                    // grad_self = out.grad @ other.T
                    for (size_t i = 0; i < M; ++i) {
                        for (size_t k = 0; k < K; ++k) {
                            for (size_t j = 0; j < N; ++j) {
                                self->grad[i * K + k] += out->grad[i * N + j] * other->data[k * N + j];
                            }
                        }
                    }
                }
                if (other->requires_grad) {
                    // grad_other = self.T @ out.grad
                    for (size_t k = 0; k < K; ++k) {
                        for (size_t j = 0; j < N; ++j) {
                            for (size_t i = 0; i < M; ++i) {
                                other->grad[k * N + j] += self->data[i * K + k] * out->grad[i * N + j];
                            }
                        }
                    }
                }
            };
        }
        return out;
    }
    
    // Activation functions
    TensorPtr relu() const {
        auto out = std::make_shared<Tensor>(data, shape, requires_grad);
        for (auto& v : out->data) v = std::max(0.0f, v);
        
        if (requires_grad) {
            out->grad.resize(out->size(), 0.0f);
            auto self_ptr = std::const_pointer_cast<Tensor>(shared_from_this());
            out->prev_ = {self_ptr};
            out->op_ = "relu";
            auto out_weak = std::weak_ptr<Tensor>(out);
            auto self_weak = std::weak_ptr<Tensor>(self_ptr);
            
            out->backward_fn_ = [out_weak, self_weak]() {
                auto out = out_weak.lock();
                auto self = self_weak.lock();
                if (!out || !self) return;
                for (size_t i = 0; i < self->size(); ++i) {
                    self->grad[i] += (self->data[i] > 0 ? 1.0f : 0.0f) * out->grad[i];
                }
            };
        }
        return out;
    }
    
    TensorPtr sigmoid() const {
        auto out = std::make_shared<Tensor>(data, shape, requires_grad);
        for (auto& v : out->data) {
            v = 1.0f / (1.0f + std::exp(-std::clamp(v, -88.0f, 88.0f)));
        }
        
        if (requires_grad) {
            out->grad.resize(out->size(), 0.0f);
            auto self_ptr = std::const_pointer_cast<Tensor>(shared_from_this());
            out->prev_ = {self_ptr};
            out->op_ = "sigmoid";
            auto out_weak = std::weak_ptr<Tensor>(out);
            auto self_weak = std::weak_ptr<Tensor>(self_ptr);
            
            out->backward_fn_ = [out_weak, self_weak]() {
                auto out = out_weak.lock();
                auto self = self_weak.lock();
                if (!out || !self) return;
                for (size_t i = 0; i < self->size(); ++i) {
                    float s = out->data[i];
                    self->grad[i] += s * (1.0f - s) * out->grad[i];
                }
            };
        }
        return out;
    }
    
    TensorPtr tanh_() const {
        auto out = std::make_shared<Tensor>(data, shape, requires_grad);
        for (auto& v : out->data) v = std::tanh(v);
        
        if (requires_grad) {
            out->grad.resize(out->size(), 0.0f);
            auto self_ptr = std::const_pointer_cast<Tensor>(shared_from_this());
            out->prev_ = {self_ptr};
            out->op_ = "tanh";
            auto out_weak = std::weak_ptr<Tensor>(out);
            auto self_weak = std::weak_ptr<Tensor>(self_ptr);
            
            out->backward_fn_ = [out_weak, self_weak]() {
                auto out = out_weak.lock();
                auto self = self_weak.lock();
                if (!out || !self) return;
                for (size_t i = 0; i < self->size(); ++i) {
                    float t = out->data[i];
                    self->grad[i] += (1.0f - t * t) * out->grad[i];
                }
            };
        }
        return out;
    }
    
    TensorPtr exp_() const {
        auto out = std::make_shared<Tensor>(data, shape, requires_grad);
        for (auto& v : out->data) v = std::exp(std::clamp(v, -88.0f, 88.0f));
        
        if (requires_grad) {
            out->grad.resize(out->size(), 0.0f);
            auto self_ptr = std::const_pointer_cast<Tensor>(shared_from_this());
            out->prev_ = {self_ptr};
            out->op_ = "exp";
            auto out_weak = std::weak_ptr<Tensor>(out);
            auto self_weak = std::weak_ptr<Tensor>(self_ptr);
            
            out->backward_fn_ = [out_weak, self_weak]() {
                auto out = out_weak.lock();
                auto self = self_weak.lock();
                if (!out || !self) return;
                for (size_t i = 0; i < self->size(); ++i) {
                    self->grad[i] += out->data[i] * out->grad[i];
                }
            };
        }
        return out;
    }
    
    TensorPtr log_() const {
        auto out = std::make_shared<Tensor>(data, shape, requires_grad);
        for (auto& v : out->data) v = std::log(std::max(v, 1e-7f));
        
        if (requires_grad) {
            out->grad.resize(out->size(), 0.0f);
            auto self_ptr = std::const_pointer_cast<Tensor>(shared_from_this());
            out->prev_ = {self_ptr};
            out->op_ = "log";
            auto out_weak = std::weak_ptr<Tensor>(out);
            auto self_weak = std::weak_ptr<Tensor>(self_ptr);
            
            out->backward_fn_ = [out_weak, self_weak]() {
                auto out = out_weak.lock();
                auto self = self_weak.lock();
                if (!out || !self) return;
                for (size_t i = 0; i < self->size(); ++i) {
                    self->grad[i] += (1.0f / std::max(self->data[i], 1e-7f)) * out->grad[i];
                }
            };
        }
        return out;
    }
    
    // Reduction operations
    TensorPtr sum() const {
        float total = std::accumulate(data.begin(), data.end(), 0.0f);
        auto out = std::make_shared<Tensor>(total, requires_grad);
        
        if (requires_grad) {
            auto self_ptr = std::const_pointer_cast<Tensor>(shared_from_this());
            out->prev_ = {self_ptr};
            out->op_ = "sum";
            auto out_weak = std::weak_ptr<Tensor>(out);
            auto self_weak = std::weak_ptr<Tensor>(self_ptr);
            
            out->backward_fn_ = [out_weak, self_weak]() {
                auto out = out_weak.lock();
                auto self = self_weak.lock();
                if (!out || !self) return;
                for (size_t i = 0; i < self->size(); ++i) {
                    self->grad[i] += out->grad[0];
                }
            };
        }
        return out;
    }
    
    TensorPtr mean() const {
        float avg = std::accumulate(data.begin(), data.end(), 0.0f) / size();
        auto out = std::make_shared<Tensor>(avg, requires_grad);
        
        if (requires_grad) {
            auto self_ptr = std::const_pointer_cast<Tensor>(shared_from_this());
            out->prev_ = {self_ptr};
            out->op_ = "mean";
            auto out_weak = std::weak_ptr<Tensor>(out);
            auto self_weak = std::weak_ptr<Tensor>(self_ptr);
            size_t n = size();
            
            out->backward_fn_ = [out_weak, self_weak, n]() {
                auto out = out_weak.lock();
                auto self = self_weak.lock();
                if (!out || !self) return;
                float factor = 1.0f / n;
                for (size_t i = 0; i < self->size(); ++i) {
                    self->grad[i] += factor * out->grad[0];
                }
            };
        }
        return out;
    }
    
    TensorPtr max() const {
        auto max_it = std::max_element(data.begin(), data.end());
        float max_val = *max_it;
        size_t max_idx = std::distance(data.begin(), max_it);
        
        auto out = std::make_shared<Tensor>(max_val, requires_grad);
        
        if (requires_grad) {
            auto self_ptr = std::const_pointer_cast<Tensor>(shared_from_this());
            out->prev_ = {self_ptr};
            out->op_ = "max";
            auto out_weak = std::weak_ptr<Tensor>(out);
            auto self_weak = std::weak_ptr<Tensor>(self_ptr);
            
            out->backward_fn_ = [out_weak, self_weak, max_idx]() {
                auto out = out_weak.lock();
                auto self = self_weak.lock();
                if (!out || !self) return;
                self->grad[max_idx] += out->grad[0];
            };
        }
        return out;
    }
    
    TensorPtr min() const {
        auto min_it = std::min_element(data.begin(), data.end());
        float min_val = *min_it;
        size_t min_idx = std::distance(data.begin(), min_it);
        
        auto out = std::make_shared<Tensor>(min_val, requires_grad);
        
        if (requires_grad) {
            auto self_ptr = std::const_pointer_cast<Tensor>(shared_from_this());
            out->prev_ = {self_ptr};
            out->op_ = "min";
            auto out_weak = std::weak_ptr<Tensor>(out);
            auto self_weak = std::weak_ptr<Tensor>(self_ptr);
            
            out->backward_fn_ = [out_weak, self_weak, min_idx]() {
                auto out = out_weak.lock();
                auto self = self_weak.lock();
                if (!out || !self) return;
                self->grad[min_idx] += out->grad[0];
            };
        }
        return out;
    }
    
    // Backpropagation
    void backward() {
        if (!requires_grad) {
            throw std::runtime_error("Cannot call backward on tensor that doesn't require grad");
        }
        
        // Build topological order
        std::vector<Tensor*> topo;
        std::set<Tensor*> visited;
        
        std::function<void(Tensor*)> build_topo = [&](Tensor* v) {
            if (visited.count(v) == 0) {
                visited.insert(v);
                for (auto& child : v->prev_) {
                    build_topo(child.get());
                }
                topo.push_back(v);
            }
        };
        
        build_topo(this);
        
        // Initialize gradient
        std::fill(grad.begin(), grad.end(), 1.0f);
        
        // Backpropagate
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if ((*it)->backward_fn_) {
                (*it)->backward_fn_();
            }
        }
    }
    
    // Zero gradients
    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
    
    // Detach from computational graph
    TensorPtr detach() const {
        return std::make_shared<Tensor>(data, shape, false);
    }
    
    // Clone
    TensorPtr clone() const {
        auto out = std::make_shared<Tensor>(data, shape, requires_grad);
        if (requires_grad) out->grad = grad;
        return out;
    }
    
private:
    size_t flatIndex(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size()) {
            throw std::runtime_error("Index dimension mismatch");
        }
        size_t idx = 0;
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            idx += indices[i] * stride;
            stride *= shape[i];
        }
        return idx;
    }
    
    std::vector<size_t> broadcastShape(const std::vector<size_t>& a, const std::vector<size_t>& b) const {
        size_t ndims = std::max(a.size(), b.size());
        std::vector<size_t> result(ndims);
        for (size_t i = 0; i < ndims; ++i) {
            size_t da = i < a.size() ? a[a.size() - 1 - i] : 1;
            size_t db = i < b.size() ? b[b.size() - 1 - i] : 1;
            if (da != db && da != 1 && db != 1) {
                throw std::runtime_error("Cannot broadcast shapes");
            }
            result[ndims - 1 - i] = std::max(da, db);
        }
        return result;
    }
    
    float getBroadcast(size_t flat_idx, const std::vector<size_t>& target_shape) const {
        std::vector<size_t> indices(target_shape.size());
        size_t tmp = flat_idx;
        for (int i = target_shape.size() - 1; i >= 0; --i) {
            indices[i] = tmp % target_shape[i];
            tmp /= target_shape[i];
        }
        
        // Map to source indices
        size_t offset = target_shape.size() - shape.size();
        size_t src_idx = 0;
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            size_t idx = indices[i + offset];
            if (shape[i] == 1) idx = 0;
            src_idx += idx * stride;
            stride *= shape[i];
        }
        return data[src_idx];
    }
    
    size_t broadcastIndex(size_t flat_idx, const std::vector<size_t>& target_shape) const {
        std::vector<size_t> indices(target_shape.size());
        size_t tmp = flat_idx;
        for (int i = target_shape.size() - 1; i >= 0; --i) {
            indices[i] = tmp % target_shape[i];
            tmp /= target_shape[i];
        }
        
        size_t offset = target_shape.size() - shape.size();
        size_t src_idx = 0;
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            size_t idx = indices[i + offset];
            if (shape[i] == 1) idx = 0;
            src_idx += idx * stride;
            stride *= shape[i];
        }
        return src_idx;
    }
};

// Operator overloads
inline TensorPtr operator+(const TensorPtr& a, const TensorPtr& b) { return a->add(b); }
inline TensorPtr operator-(const TensorPtr& a, const TensorPtr& b) { return a->sub(b); }
inline TensorPtr operator*(const TensorPtr& a, const TensorPtr& b) { return a->mul(b); }
inline TensorPtr operator/(const TensorPtr& a, const TensorPtr& b) { return a->div(b); }

/**
 * @brief Parameter - learnable parameter (tensor with gradient)
 */
class Parameter {
public:
    TensorPtr data;
    
    Parameter() = default;
    
    explicit Parameter(const TensorPtr& tensor) : data(tensor) {
        data->requires_grad = true;
        data->grad.resize(data->size(), 0.0f);
    }
    
    Parameter(const std::vector<float>& values, const std::vector<size_t>& shape)
        : data(Tensor::create(values, shape, true)) {}
    
    void zero_grad() {
        if (data) data->zero_grad();
    }
    
    TensorPtr operator->() { return data; }
    const TensorPtr operator->() const { return data; }
};

} // namespace nn
} // namespace neurova
