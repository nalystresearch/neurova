// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file nn.hpp
 * @brief Neural Network module for Neurova
 * 
 * This header provides a complete neural network framework including:
 * - Tensor operations with automatic differentiation
 * - Layer modules (Linear, Conv, Pooling, Normalization, etc.)
 * - Activation functions
 * - Loss functions
 * - Optimizers and learning rate schedulers
 * - Attention mechanisms and Transformers
 * - Recurrent networks (RNN, LSTM, GRU)
 * - Embedding layers
 * - Functional API
 */

#pragma once

// Core components
#include "nn/tensor.hpp"
#include "nn/layers.hpp"

// Layer types
#include "nn/activation.hpp"
#include "nn/linear.hpp"
#include "nn/conv.hpp"
#include "nn/pooling.hpp"
#include "nn/normalization.hpp"
#include "nn/dropout.hpp"
#include "nn/padding.hpp"
#include "nn/embedding.hpp"

// Recurrent and attention
#include "nn/recurrent.hpp"
#include "nn/attention.hpp"

// Training components
#include "nn/loss.hpp"
#include "nn/optim.hpp"
#include "nn/scheduler.hpp"

// Functional API
#include "nn/functional.hpp"

namespace neurova {
namespace nn {

/**
 * @brief Model utilities
 */

/**
 * @brief Count total parameters in a module
 */
inline size_t count_parameters(const Module& module) {
    size_t total = 0;
    for (const auto& [name, param] : module.named_parameters()) {
        total += param.data().numel();
    }
    return total;
}

/**
 * @brief Count trainable parameters
 */
inline size_t count_trainable_parameters(const Module& module) {
    size_t total = 0;
    for (const auto& [name, param] : module.named_parameters()) {
        if (param.requires_grad()) {
            total += param.data().numel();
        }
    }
    return total;
}

/**
 * @brief Save model state to vector (simplified serialization)
 */
inline std::vector<float> save_state_dict(const Module& module) {
    std::vector<float> state;
    for (const auto& [name, param] : module.named_parameters()) {
        for (float v : param.data().data()) {
            state.push_back(v);
        }
    }
    return state;
}

/**
 * @brief Load model state from vector
 */
inline void load_state_dict(Module& module, const std::vector<float>& state) {
    size_t offset = 0;
    for (auto& [name, param] : module.named_parameters()) {
        size_t size = param.data().numel();
        std::vector<float> data(state.begin() + offset, state.begin() + offset + size);
        param.data() = Tensor(data, param.data().shape());
        offset += size;
    }
}

/**
 * @brief Clip gradient norm
 */
inline float clip_grad_norm(std::vector<Parameter*>& parameters, float max_norm, float norm_type = 2.0f) {
    float total_norm = 0.0f;
    
    for (auto* param : parameters) {
        if (param->requires_grad() && param->grad().numel() > 0) {
            for (float g : param->grad().data()) {
                total_norm += std::pow(std::abs(g), norm_type);
            }
        }
    }
    total_norm = std::pow(total_norm, 1.0f / norm_type);
    
    float clip_coef = max_norm / (total_norm + 1e-6f);
    if (clip_coef < 1.0f) {
        for (auto* param : parameters) {
            if (param->requires_grad() && param->grad().numel() > 0) {
                std::vector<float> clipped(param->grad().numel());
                for (size_t i = 0; i < param->grad().numel(); ++i) {
                    clipped[i] = param->grad().data()[i] * clip_coef;
                }
                param->grad() = Tensor(clipped, param->grad().shape());
            }
        }
    }
    
    return total_norm;
}

/**
 * @brief Clip gradient value
 */
inline void clip_grad_value(std::vector<Parameter*>& parameters, float clip_value) {
    for (auto* param : parameters) {
        if (param->requires_grad() && param->grad().numel() > 0) {
            std::vector<float> clipped(param->grad().numel());
            for (size_t i = 0; i < param->grad().numel(); ++i) {
                clipped[i] = std::max(-clip_value, std::min(clip_value, param->grad().data()[i]));
            }
            param->grad() = Tensor(clipped, param->grad().shape());
        }
    }
}

/**
 * @brief Initialize weights with Xavier/Glorot uniform
 */
inline void xavier_uniform_(Tensor& tensor, float gain = 1.0f) {
    auto shape = tensor.shape();
    int fan_in = shape.size() > 1 ? shape[1] : shape[0];
    int fan_out = shape[0];
    
    float std = gain * std::sqrt(2.0f / (fan_in + fan_out));
    float a = std::sqrt(3.0f) * std;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-a, a);
    
    std::vector<float> data(tensor.numel());
    for (auto& v : data) {
        v = dist(gen);
    }
    tensor = Tensor(data, shape);
}

/**
 * @brief Initialize weights with Xavier/Glorot normal
 */
inline void xavier_normal_(Tensor& tensor, float gain = 1.0f) {
    auto shape = tensor.shape();
    int fan_in = shape.size() > 1 ? shape[1] : shape[0];
    int fan_out = shape[0];
    
    float std = gain * std::sqrt(2.0f / (fan_in + fan_out));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std);
    
    std::vector<float> data(tensor.numel());
    for (auto& v : data) {
        v = dist(gen);
    }
    tensor = Tensor(data, shape);
}

/**
 * @brief Initialize weights with Kaiming/He uniform
 */
inline void kaiming_uniform_(Tensor& tensor, float a = 0.0f, const std::string& mode = "fan_in") {
    auto shape = tensor.shape();
    int fan = (mode == "fan_in") ? (shape.size() > 1 ? shape[1] : shape[0]) : shape[0];
    
    float gain = std::sqrt(2.0f / (1.0f + a * a));
    float std = gain / std::sqrt(static_cast<float>(fan));
    float bound = std::sqrt(3.0f) * std;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-bound, bound);
    
    std::vector<float> data(tensor.numel());
    for (auto& v : data) {
        v = dist(gen);
    }
    tensor = Tensor(data, shape);
}

/**
 * @brief Initialize weights with Kaiming/He normal
 */
inline void kaiming_normal_(Tensor& tensor, float a = 0.0f, const std::string& mode = "fan_in") {
    auto shape = tensor.shape();
    int fan = (mode == "fan_in") ? (shape.size() > 1 ? shape[1] : shape[0]) : shape[0];
    
    float gain = std::sqrt(2.0f / (1.0f + a * a));
    float std = gain / std::sqrt(static_cast<float>(fan));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std);
    
    std::vector<float> data(tensor.numel());
    for (auto& v : data) {
        v = dist(gen);
    }
    tensor = Tensor(data, shape);
}

/**
 * @brief Initialize tensor with orthogonal values
 */
inline void orthogonal_(Tensor& tensor, float gain = 1.0f) {
    auto shape = tensor.shape();
    int rows = shape[0];
    int cols = tensor.numel() / rows;
    
    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> data(tensor.numel());
    for (auto& v : data) {
        v = dist(gen) * gain;
    }
    
    // Note: Full QR decomposition would be needed for true orthogonal init
    // This is a simplified version
    tensor = Tensor(data, shape);
}

} // namespace nn
} // namespace neurova
