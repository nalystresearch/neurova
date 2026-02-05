// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file dropout.hpp
 * @brief Dropout regularization layers
 * 
 * Neurova implementation of dropout for regularization.
 */

#pragma once

#include "layers.hpp"
#include <random>

namespace neurova {
namespace nn {

/**
 * @brief Standard Dropout
 */
class Dropout : public Module {
private:
    float p_;
    bool inplace_;
    
public:
    explicit Dropout(float p = 0.5f, bool inplace = false)
        : p_(p), inplace_(inplace) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        if (!training_ || p_ == 0.0f) {
            return x;
        }
        
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        float scale = 1.0f / (1.0f - p_);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(1.0 - p_);
        
        for (auto& v : out->data) {
            if (dist(gen)) {
                v *= scale;
            } else {
                v = 0.0f;
            }
        }
        
        return out;
    }
};

/**
 * @brief 2D Dropout (drops entire channels)
 */
class Dropout2d : public Module {
private:
    float p_;
    bool inplace_;
    
public:
    explicit Dropout2d(float p = 0.5f, bool inplace = false)
        : p_(p), inplace_(inplace) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        if (!training_ || p_ == 0.0f) {
            return x;
        }
        
        // x: (batch, channels, height, width)
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t spatial = x->shape[2] * x->shape[3];
        
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        float scale = 1.0f / (1.0f - p_);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(1.0 - p_);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                bool keep = dist(gen);
                for (size_t s = 0; s < spatial; ++s) {
                    size_t idx = b * channels * spatial + c * spatial + s;
                    if (keep) {
                        out->data[idx] *= scale;
                    } else {
                        out->data[idx] = 0.0f;
                    }
                }
            }
        }
        
        return out;
    }
};

/**
 * @brief 3D Dropout (drops entire channels)
 */
class Dropout3d : public Module {
private:
    float p_;
    bool inplace_;
    
public:
    explicit Dropout3d(float p = 0.5f, bool inplace = false)
        : p_(p), inplace_(inplace) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        if (!training_ || p_ == 0.0f) {
            return x;
        }
        
        // x: (batch, channels, depth, height, width)
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t spatial = x->size() / (batch * channels);
        
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        float scale = 1.0f / (1.0f - p_);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(1.0 - p_);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                bool keep = dist(gen);
                for (size_t s = 0; s < spatial; ++s) {
                    size_t idx = b * channels * spatial + c * spatial + s;
                    if (keep) {
                        out->data[idx] *= scale;
                    } else {
                        out->data[idx] = 0.0f;
                    }
                }
            }
        }
        
        return out;
    }
};

/**
 * @brief Alpha Dropout (for SELU)
 */
class AlphaDropout : public Module {
private:
    float p_;
    bool inplace_;
    static constexpr float alpha_ = -1.7580993408473766f;
    
public:
    explicit AlphaDropout(float p = 0.5f, bool inplace = false)
        : p_(p), inplace_(inplace) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        if (!training_ || p_ == 0.0f) {
            return x;
        }
        
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        
        // Compute affine transformation parameters
        float a = 1.0f / std::sqrt((1.0f - p_) + p_ * alpha_ * alpha_);
        float b = -a * alpha_ * p_;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(1.0 - p_);
        
        for (auto& v : out->data) {
            if (!dist(gen)) {
                v = alpha_;
            }
            v = a * v + b;
        }
        
        return out;
    }
};

/**
 * @brief Feature Alpha Dropout
 */
class FeatureAlphaDropout : public Module {
private:
    float p_;
    
public:
    explicit FeatureAlphaDropout(float p = 0.5f) : p_(p) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        if (!training_ || p_ == 0.0f) {
            return x;
        }
        
        // Similar to AlphaDropout but for features
        return AlphaDropout(p_).forward(x);
    }
};

} // namespace nn
} // namespace neurova
