// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file activation.hpp
 * @brief Activation functions for neural networks
 * 
 * Neurova implementation of common activation functions.
 */

#pragma once

#include "layers.hpp"
#include <cmath>
#include <algorithm>

namespace neurova {
namespace nn {

/**
 * @brief ReLU activation: max(0, x)
 */
class ReLU : public Module {
private:
    bool inplace_;
    
public:
    explicit ReLU(bool inplace = false) : inplace_(inplace) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        return x->relu();
    }
};

/**
 * @brief Leaky ReLU activation
 */
class LeakyReLU : public Module {
private:
    float negative_slope_;
    bool inplace_;
    
public:
    explicit LeakyReLU(float negative_slope = 0.01f, bool inplace = false)
        : negative_slope_(negative_slope), inplace_(inplace) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        for (auto& v : out->data) {
            v = v >= 0 ? v : negative_slope_ * v;
        }
        return out;
    }
};

/**
 * @brief PReLU - Parametric ReLU
 */
class PReLU : public Module {
private:
    Parameter weight_;
    int num_parameters_;
    
public:
    explicit PReLU(int num_parameters = 1, float init = 0.25f)
        : num_parameters_(num_parameters) {
        std::vector<float> w(num_parameters, init);
        weight_ = Parameter(Tensor::create(w, {static_cast<size_t>(num_parameters)}, true));
        register_parameter("weight", weight_);
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        for (size_t i = 0; i < out->data.size(); ++i) {
            float a = weight_.data->data[i % num_parameters_];
            out->data[i] = out->data[i] >= 0 ? out->data[i] : a * out->data[i];
        }
        return out;
    }
};

/**
 * @brief ELU activation: x if x > 0, else alpha * (exp(x) - 1)
 */
class ELU : public Module {
private:
    float alpha_;
    bool inplace_;
    
public:
    explicit ELU(float alpha = 1.0f, bool inplace = false)
        : alpha_(alpha), inplace_(inplace) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        for (auto& v : out->data) {
            v = v > 0 ? v : alpha_ * (std::exp(v) - 1);
        }
        return out;
    }
};

/**
 * @brief SELU - Scaled Exponential Linear Unit
 */
class SELU : public Module {
private:
    bool inplace_;
    static constexpr float alpha_ = 1.6732632423543772f;
    static constexpr float scale_ = 1.0507009873554804f;
    
public:
    explicit SELU(bool inplace = false) : inplace_(inplace) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        for (auto& v : out->data) {
            v = scale_ * (v > 0 ? v : alpha_ * (std::exp(v) - 1));
        }
        return out;
    }
};

/**
 * @brief GELU - Gaussian Error Linear Unit
 */
class GELU : public Module {
private:
    std::string approximate_;
    
public:
    explicit GELU(const std::string& approximate = "none") : approximate_(approximate) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        
        if (approximate_ == "tanh") {
            // Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const float sqrt_2_pi = std::sqrt(2.0f / 3.14159265f);
            for (auto& v : out->data) {
                float inner = sqrt_2_pi * (v + 0.044715f * v * v * v);
                v = 0.5f * v * (1.0f + std::tanh(inner));
            }
        } else {
            // Exact: 0.5 * x * (1 + erf(x / sqrt(2)))
            const float sqrt_2 = std::sqrt(2.0f);
            for (auto& v : out->data) {
                v = 0.5f * v * (1.0f + std::erf(v / sqrt_2));
            }
        }
        return out;
    }
};

/**
 * @brief Sigmoid activation: 1 / (1 + exp(-x))
 */
class Sigmoid : public Module {
public:
    TensorPtr forward(const TensorPtr& x) override {
        return x->sigmoid();
    }
};

/**
 * @brief Tanh activation
 */
class Tanh : public Module {
public:
    TensorPtr forward(const TensorPtr& x) override {
        return x->tanh_();
    }
};

/**
 * @brief Softmax activation
 */
class Softmax : public Module {
private:
    int dim_;
    
public:
    explicit Softmax(int dim = -1) : dim_(dim) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        // For simplicity, implement for 2D tensor along last dimension
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        
        if (x->shape.size() == 2) {
            size_t batch = x->shape[0];
            size_t features = x->shape[1];
            
            for (size_t b = 0; b < batch; ++b) {
                // Find max for numerical stability
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < features; ++i) {
                    max_val = std::max(max_val, x->data[b * features + i]);
                }
                
                // Compute exp and sum
                float sum = 0.0f;
                for (size_t i = 0; i < features; ++i) {
                    out->data[b * features + i] = std::exp(x->data[b * features + i] - max_val);
                    sum += out->data[b * features + i];
                }
                
                // Normalize
                for (size_t i = 0; i < features; ++i) {
                    out->data[b * features + i] /= sum;
                }
            }
        } else {
            // Simple 1D softmax
            float max_val = *std::max_element(x->data.begin(), x->data.end());
            float sum = 0.0f;
            for (size_t i = 0; i < out->data.size(); ++i) {
                out->data[i] = std::exp(x->data[i] - max_val);
                sum += out->data[i];
            }
            for (auto& v : out->data) v /= sum;
        }
        
        return out;
    }
};

/**
 * @brief LogSoftmax activation
 */
class LogSoftmax : public Module {
private:
    int dim_;
    
public:
    explicit LogSoftmax(int dim = -1) : dim_(dim) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        
        if (x->shape.size() == 2) {
            size_t batch = x->shape[0];
            size_t features = x->shape[1];
            
            for (size_t b = 0; b < batch; ++b) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < features; ++i) {
                    max_val = std::max(max_val, x->data[b * features + i]);
                }
                
                float sum = 0.0f;
                for (size_t i = 0; i < features; ++i) {
                    sum += std::exp(x->data[b * features + i] - max_val);
                }
                float log_sum = max_val + std::log(sum);
                
                for (size_t i = 0; i < features; ++i) {
                    out->data[b * features + i] = x->data[b * features + i] - log_sum;
                }
            }
        } else {
            float max_val = *std::max_element(x->data.begin(), x->data.end());
            float sum = 0.0f;
            for (auto v : x->data) sum += std::exp(v - max_val);
            float log_sum = max_val + std::log(sum);
            for (size_t i = 0; i < out->data.size(); ++i) {
                out->data[i] = x->data[i] - log_sum;
            }
        }
        
        return out;
    }
};

/**
 * @brief SiLU/Swish activation: x * sigmoid(x)
 */
class SiLU : public Module {
private:
    bool inplace_;
    
public:
    explicit SiLU(bool inplace = false) : inplace_(inplace) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto sig = x->sigmoid();
        return x->mul(sig);
    }
};

/**
 * @brief Mish activation: x * tanh(softplus(x))
 */
class Mish : public Module {
private:
    bool inplace_;
    
public:
    explicit Mish(bool inplace = false) : inplace_(inplace) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        for (size_t i = 0; i < out->data.size(); ++i) {
            float sp = std::log(1.0f + std::exp(x->data[i]));
            out->data[i] = x->data[i] * std::tanh(sp);
        }
        return out;
    }
};

/**
 * @brief Softplus activation: log(1 + exp(x))
 */
class Softplus : public Module {
private:
    float beta_;
    float threshold_;
    
public:
    Softplus(float beta = 1.0f, float threshold = 20.0f)
        : beta_(beta), threshold_(threshold) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        for (auto& v : out->data) {
            float scaled = beta_ * v;
            v = scaled > threshold_ ? v : std::log(1.0f + std::exp(scaled)) / beta_;
        }
        return out;
    }
};

/**
 * @brief Softsign activation: x / (1 + |x|)
 */
class Softsign : public Module {
public:
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        for (auto& v : out->data) {
            v = v / (1.0f + std::abs(v));
        }
        return out;
    }
};

/**
 * @brief Hardtanh - piecewise linear approximation of tanh
 */
class Hardtanh : public Module {
private:
    float min_val_;
    float max_val_;
    bool inplace_;
    
public:
    Hardtanh(float min_val = -1.0f, float max_val = 1.0f, bool inplace = false)
        : min_val_(min_val), max_val_(max_val), inplace_(inplace) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        for (auto& v : out->data) {
            v = std::clamp(v, min_val_, max_val_);
        }
        return out;
    }
};

/**
 * @brief ReLU6 - ReLU capped at 6
 */
class ReLU6 : public Module {
private:
    bool inplace_;
    
public:
    explicit ReLU6(bool inplace = false) : inplace_(inplace) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        for (auto& v : out->data) {
            v = std::clamp(v, 0.0f, 6.0f);
        }
        return out;
    }
};

/**
 * @brief Hardsigmoid: ReLU6(x+3) / 6
 */
class Hardsigmoid : public Module {
private:
    bool inplace_;
    
public:
    explicit Hardsigmoid(bool inplace = false) : inplace_(inplace) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        for (auto& v : out->data) {
            v = std::clamp(v + 3.0f, 0.0f, 6.0f) / 6.0f;
        }
        return out;
    }
};

/**
 * @brief Hardswish: x * ReLU6(x+3) / 6
 */
class Hardswish : public Module {
private:
    bool inplace_;
    
public:
    explicit Hardswish(bool inplace = false) : inplace_(inplace) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        for (auto& v : out->data) {
            v = v * std::clamp(v + 3.0f, 0.0f, 6.0f) / 6.0f;
        }
        return out;
    }
};

/**
 * @brief Softshrink activation
 */
class Softshrink : public Module {
private:
    float lambda_;
    
public:
    explicit Softshrink(float lambda = 0.5f) : lambda_(lambda) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        for (auto& v : out->data) {
            if (v > lambda_) v = v - lambda_;
            else if (v < -lambda_) v = v + lambda_;
            else v = 0.0f;
        }
        return out;
    }
};

/**
 * @brief Hardshrink activation
 */
class Hardshrink : public Module {
private:
    float lambda_;
    
public:
    explicit Hardshrink(float lambda = 0.5f) : lambda_(lambda) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        for (auto& v : out->data) {
            v = (std::abs(v) > lambda_) ? v : 0.0f;
        }
        return out;
    }
};

/**
 * @brief Tanhshrink activation: x - tanh(x)
 */
class Tanhshrink : public Module {
public:
    TensorPtr forward(const TensorPtr& x) override {
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        for (auto& v : out->data) {
            v = v - std::tanh(v);
        }
        return out;
    }
};

/**
 * @brief GLU - Gated Linear Unit
 */
class GLU : public Module {
private:
    int dim_;
    
public:
    explicit GLU(int dim = -1) : dim_(dim) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        // Split input in half along dim and apply sigmoid gate
        size_t half_size = x->size() / 2;
        std::vector<float> a(x->data.begin(), x->data.begin() + half_size);
        std::vector<float> b(x->data.begin() + half_size, x->data.end());
        
        // a * sigmoid(b)
        std::vector<float> result(half_size);
        for (size_t i = 0; i < half_size; ++i) {
            float sig = 1.0f / (1.0f + std::exp(-b[i]));
            result[i] = a[i] * sig;
        }
        
        auto new_shape = x->shape;
        new_shape[new_shape.size() - 1] /= 2;
        return std::make_shared<Tensor>(result, new_shape, x->requires_grad);
    }
};

} // namespace nn
} // namespace neurova
