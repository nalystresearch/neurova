// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file linear.hpp
 * @brief Linear (fully connected) layers
 * 
 * Neurova implementation of linear layers.
 */

#pragma once

#include "layers.hpp"
#include <cmath>

namespace neurova {
namespace nn {

/**
 * @brief Linear (fully connected) layer: y = xW^T + b
 */
class Linear : public Module {
private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;
    Parameter weight_;
    Parameter bias_;
    
public:
    /**
     * @brief Construct Linear layer
     * @param in_features Size of each input sample
     * @param out_features Size of each output sample
     * @param bias If true, adds a learnable bias
     */
    Linear(size_t in_features, size_t out_features, bool bias = true)
        : in_features_(in_features), out_features_(out_features), use_bias_(bias) {
        
        // Kaiming/He initialization
        float k = 1.0f / std::sqrt(static_cast<float>(in_features));
        
        // Initialize weight (out_features x in_features)
        std::vector<float> weight_data(out_features * in_features);
        for (auto& w : weight_data) {
            w = -k + 2.0f * k * static_cast<float>(rand()) / RAND_MAX;
        }
        weight_ = Parameter(Tensor::create(weight_data, {out_features, in_features}, true));
        register_parameter("weight", weight_);
        
        if (bias) {
            std::vector<float> bias_data(out_features);
            for (auto& b : bias_data) {
                b = -k + 2.0f * k * static_cast<float>(rand()) / RAND_MAX;
            }
            bias_ = Parameter(Tensor::create(bias_data, {out_features}, true));
            register_parameter("bias", bias_);
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        // y = xW^T + b
        // x: (batch, in_features), W: (out_features, in_features)
        // xW^T: (batch, out_features)
        
        auto wT = weight_.data->transpose();
        auto out = x->matmul(wT);
        
        if (use_bias_) {
            out = out->add(bias_.data);
        }
        
        return out;
    }
    
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
};

/**
 * @brief Bilinear layer: y = x1 @ W @ x2 + b
 */
class Bilinear : public Module {
private:
    size_t in1_features_;
    size_t in2_features_;
    size_t out_features_;
    bool use_bias_;
    Parameter weight_;  // (out_features, in1_features, in2_features)
    Parameter bias_;
    
public:
    Bilinear(size_t in1_features, size_t in2_features, size_t out_features, bool bias = true)
        : in1_features_(in1_features), in2_features_(in2_features), 
          out_features_(out_features), use_bias_(bias) {
        
        float k = 1.0f / std::sqrt(static_cast<float>(in1_features));
        
        std::vector<float> weight_data(out_features * in1_features * in2_features);
        for (auto& w : weight_data) {
            w = -k + 2.0f * k * static_cast<float>(rand()) / RAND_MAX;
        }
        weight_ = Parameter(Tensor::create(weight_data, 
            {out_features, in1_features, in2_features}, true));
        register_parameter("weight", weight_);
        
        if (bias) {
            std::vector<float> bias_data(out_features);
            for (auto& b : bias_data) {
                b = -k + 2.0f * k * static_cast<float>(rand()) / RAND_MAX;
            }
            bias_ = Parameter(Tensor::create(bias_data, {out_features}, true));
            register_parameter("bias", bias_);
        }
    }
    
    TensorPtr forward(const TensorPtr& x1, const TensorPtr& x2) override {
        // Simplified bilinear computation
        size_t batch = x1->shape[0];
        std::vector<float> result(batch * out_features_, 0.0f);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t o = 0; o < out_features_; ++o) {
                float sum = 0.0f;
                for (size_t i = 0; i < in1_features_; ++i) {
                    for (size_t j = 0; j < in2_features_; ++j) {
                        size_t w_idx = o * in1_features_ * in2_features_ + i * in2_features_ + j;
                        sum += x1->data[b * in1_features_ + i] * 
                               weight_.data->data[w_idx] * 
                               x2->data[b * in2_features_ + j];
                    }
                }
                result[b * out_features_ + o] = sum;
                if (use_bias_) {
                    result[b * out_features_ + o] += bias_.data->data[o];
                }
            }
        }
        
        return std::make_shared<Tensor>(result, {batch, out_features_}, 
            x1->requires_grad || x2->requires_grad);
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("Bilinear requires two inputs");
    }
};

/**
 * @brief LazyLinear - Linear layer with lazy initialization
 */
class LazyLinear : public Module {
private:
    size_t out_features_;
    bool use_bias_;
    std::shared_ptr<Linear> linear_;
    bool initialized_ = false;
    
public:
    LazyLinear(size_t out_features, bool bias = true)
        : out_features_(out_features), use_bias_(bias) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        if (!initialized_) {
            size_t in_features = x->shape[x->shape.size() - 1];
            linear_ = std::make_shared<Linear>(in_features, out_features_, use_bias_);
            initialized_ = true;
        }
        return linear_->forward(x);
    }
};

} // namespace nn
} // namespace neurova
