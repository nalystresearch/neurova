// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file functional.hpp
 * @brief Functional API for neural network operations
 * 
 * Neurova implementation of functional operations (stateless).
 */

#pragma once

#include "tensor.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

namespace neurova {
namespace nn {
namespace functional {

// ============================================================================
// Activation Functions
// ============================================================================

inline Tensor relu(const Tensor& input) {
    return input.relu();
}

inline Tensor leaky_relu(const Tensor& input, float negative_slope = 0.01f) {
    std::vector<float> result(input.numel());
    for (size_t i = 0; i < input.numel(); ++i) {
        result[i] = input.data()[i] > 0 ? input.data()[i] : negative_slope * input.data()[i];
    }
    return Tensor(result, input.shape());
}

inline Tensor elu(const Tensor& input, float alpha = 1.0f) {
    std::vector<float> result(input.numel());
    for (size_t i = 0; i < input.numel(); ++i) {
        float x = input.data()[i];
        result[i] = x > 0 ? x : alpha * (std::exp(x) - 1.0f);
    }
    return Tensor(result, input.shape());
}

inline Tensor selu(const Tensor& input) {
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;
    
    std::vector<float> result(input.numel());
    for (size_t i = 0; i < input.numel(); ++i) {
        float x = input.data()[i];
        result[i] = scale * (x > 0 ? x : alpha * (std::exp(x) - 1.0f));
    }
    return Tensor(result, input.shape());
}

inline Tensor gelu(const Tensor& input, bool approximate = true) {
    return input.gelu();
}

inline Tensor sigmoid(const Tensor& input) {
    return input.sigmoid();
}

inline Tensor tanh(const Tensor& input) {
    return input.tanh();
}

inline Tensor softmax(const Tensor& input, int dim = -1) {
    auto shape = input.shape();
    if (dim < 0) dim += static_cast<int>(shape.size());
    
    int dim_size = shape[dim];
    int outer_size = 1, inner_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= shape[i];
    for (int i = dim + 1; i < static_cast<int>(shape.size()); ++i) inner_size *= shape[i];
    
    std::vector<float> result(input.numel());
    
    for (int outer = 0; outer < outer_size; ++outer) {
        for (int inner = 0; inner < inner_size; ++inner) {
            // Find max for stability
            float max_val = -std::numeric_limits<float>::infinity();
            for (int d = 0; d < dim_size; ++d) {
                int idx = outer * dim_size * inner_size + d * inner_size + inner;
                max_val = std::max(max_val, input.data()[idx]);
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (int d = 0; d < dim_size; ++d) {
                int idx = outer * dim_size * inner_size + d * inner_size + inner;
                result[idx] = std::exp(input.data()[idx] - max_val);
                sum += result[idx];
            }
            
            // Normalize
            for (int d = 0; d < dim_size; ++d) {
                int idx = outer * dim_size * inner_size + d * inner_size + inner;
                result[idx] /= sum;
            }
        }
    }
    
    return Tensor(result, shape);
}

inline Tensor log_softmax(const Tensor& input, int dim = -1) {
    Tensor sm = softmax(input, dim);
    return sm.log();
}

inline Tensor silu(const Tensor& input) {
    return input * input.sigmoid();
}

inline Tensor mish(const Tensor& input) {
    std::vector<float> result(input.numel());
    for (size_t i = 0; i < input.numel(); ++i) {
        float x = input.data()[i];
        float sp = std::log(1.0f + std::exp(x));  // softplus
        result[i] = x * std::tanh(sp);
    }
    return Tensor(result, input.shape());
}

inline Tensor hardtanh(const Tensor& input, float min_val = -1.0f, float max_val = 1.0f) {
    std::vector<float> result(input.numel());
    for (size_t i = 0; i < input.numel(); ++i) {
        result[i] = std::max(min_val, std::min(max_val, input.data()[i]));
    }
    return Tensor(result, input.shape());
}

inline Tensor relu6(const Tensor& input) {
    return hardtanh(input, 0.0f, 6.0f);
}

// ============================================================================
// Loss Functions
// ============================================================================

inline Tensor mse_loss(const Tensor& input, const Tensor& target, 
                       const std::string& reduction = "mean") {
    Tensor diff = input - target;
    Tensor squared = diff * diff;
    
    if (reduction == "mean") {
        return squared.mean();
    } else if (reduction == "sum") {
        return squared.sum();
    }
    return squared;
}

inline Tensor l1_loss(const Tensor& input, const Tensor& target,
                      const std::string& reduction = "mean") {
    std::vector<float> abs_diff(input.numel());
    for (size_t i = 0; i < input.numel(); ++i) {
        abs_diff[i] = std::abs(input.data()[i] - target.data()[i]);
    }
    Tensor loss(abs_diff, input.shape());
    
    if (reduction == "mean") {
        return loss.mean();
    } else if (reduction == "sum") {
        return loss.sum();
    }
    return loss;
}

inline Tensor smooth_l1_loss(const Tensor& input, const Tensor& target,
                             const std::string& reduction = "mean", float beta = 1.0f) {
    std::vector<float> result(input.numel());
    
    for (size_t i = 0; i < input.numel(); ++i) {
        float diff = std::abs(input.data()[i] - target.data()[i]);
        if (diff < beta) {
            result[i] = 0.5f * diff * diff / beta;
        } else {
            result[i] = diff - 0.5f * beta;
        }
    }
    
    Tensor loss(result, input.shape());
    
    if (reduction == "mean") {
        return loss.mean();
    } else if (reduction == "sum") {
        return loss.sum();
    }
    return loss;
}

inline Tensor cross_entropy(const Tensor& input, const Tensor& target,
                            const std::string& reduction = "mean",
                            float label_smoothing = 0.0f) {
    Tensor log_sm = log_softmax(input, -1);
    
    std::vector<float> loss_values;
    int num_classes = input.shape().back();
    int batch_size = input.numel() / num_classes;
    
    for (int i = 0; i < batch_size; ++i) {
        int label = static_cast<int>(target.data()[i]);
        
        if (label_smoothing > 0.0f) {
            float smooth_loss = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                float target_prob = (c == label) ? (1.0f - label_smoothing) : 0.0f;
                target_prob += label_smoothing / num_classes;
                smooth_loss -= target_prob * log_sm.data()[i * num_classes + c];
            }
            loss_values.push_back(smooth_loss);
        } else {
            loss_values.push_back(-log_sm.data()[i * num_classes + label]);
        }
    }
    
    Tensor loss(loss_values, {batch_size});
    
    if (reduction == "mean") {
        return loss.mean();
    } else if (reduction == "sum") {
        return loss.sum();
    }
    return loss;
}

inline Tensor nll_loss(const Tensor& input, const Tensor& target,
                       const std::string& reduction = "mean") {
    int num_classes = input.shape().back();
    int batch_size = input.numel() / num_classes;
    
    std::vector<float> loss_values(batch_size);
    
    for (int i = 0; i < batch_size; ++i) {
        int label = static_cast<int>(target.data()[i]);
        loss_values[i] = -input.data()[i * num_classes + label];
    }
    
    Tensor loss(loss_values, {batch_size});
    
    if (reduction == "mean") {
        return loss.mean();
    } else if (reduction == "sum") {
        return loss.sum();
    }
    return loss;
}

inline Tensor binary_cross_entropy(const Tensor& input, const Tensor& target,
                                   const std::string& reduction = "mean") {
    std::vector<float> result(input.numel());
    
    for (size_t i = 0; i < input.numel(); ++i) {
        float p = std::max(1e-7f, std::min(1.0f - 1e-7f, input.data()[i]));
        float t = target.data()[i];
        result[i] = -(t * std::log(p) + (1.0f - t) * std::log(1.0f - p));
    }
    
    Tensor loss(result, input.shape());
    
    if (reduction == "mean") {
        return loss.mean();
    } else if (reduction == "sum") {
        return loss.sum();
    }
    return loss;
}

inline Tensor binary_cross_entropy_with_logits(const Tensor& input, const Tensor& target,
                                                const std::string& reduction = "mean") {
    std::vector<float> result(input.numel());
    
    for (size_t i = 0; i < input.numel(); ++i) {
        float x = input.data()[i];
        float t = target.data()[i];
        result[i] = std::max(x, 0.0f) - x * t + std::log(1.0f + std::exp(-std::abs(x)));
    }
    
    Tensor loss(result, input.shape());
    
    if (reduction == "mean") {
        return loss.mean();
    } else if (reduction == "sum") {
        return loss.sum();
    }
    return loss;
}

// ============================================================================
// Regularization
// ============================================================================

inline Tensor dropout(const Tensor& input, float p = 0.5f, bool training = true) {
    if (!training || p == 0.0f) {
        return input;
    }
    
    std::vector<float> result(input.numel());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    float scale = 1.0f / (1.0f - p);
    
    for (size_t i = 0; i < input.numel(); ++i) {
        if (dist(gen) < p) {
            result[i] = 0.0f;
        } else {
            result[i] = input.data()[i] * scale;
        }
    }
    
    return Tensor(result, input.shape());
}

inline Tensor alpha_dropout(const Tensor& input, float p = 0.5f, bool training = true) {
    if (!training || p == 0.0f) {
        return input;
    }
    
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;
    const float alpha_p = -alpha * scale;
    
    std::vector<float> result(input.numel());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    float a = std::sqrt(p / ((1.0f - p) * std::pow(alpha_p, 2) + 1e-8f));
    float b = -a * alpha_p * p;
    
    for (size_t i = 0; i < input.numel(); ++i) {
        if (dist(gen) < p) {
            result[i] = alpha_p;
        } else {
            result[i] = input.data()[i];
        }
        result[i] = result[i] * a + b;
    }
    
    return Tensor(result, input.shape());
}

// ============================================================================
// Normalization
// ============================================================================

inline Tensor normalize(const Tensor& input, float p = 2.0f, int dim = -1, float eps = 1e-12f) {
    auto shape = input.shape();
    if (dim < 0) dim += static_cast<int>(shape.size());
    
    int dim_size = shape[dim];
    int outer_size = 1, inner_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= shape[i];
    for (int i = dim + 1; i < static_cast<int>(shape.size()); ++i) inner_size *= shape[i];
    
    std::vector<float> result(input.numel());
    
    for (int outer = 0; outer < outer_size; ++outer) {
        for (int inner = 0; inner < inner_size; ++inner) {
            // Compute p-norm
            float norm = 0.0f;
            for (int d = 0; d < dim_size; ++d) {
                int idx = outer * dim_size * inner_size + d * inner_size + inner;
                norm += std::pow(std::abs(input.data()[idx]), p);
            }
            norm = std::pow(norm, 1.0f / p);
            norm = std::max(norm, eps);
            
            // Normalize
            for (int d = 0; d < dim_size; ++d) {
                int idx = outer * dim_size * inner_size + d * inner_size + inner;
                result[idx] = input.data()[idx] / norm;
            }
        }
    }
    
    return Tensor(result, shape);
}

inline Tensor layer_norm(const Tensor& input, const std::vector<int>& normalized_shape,
                         const Tensor* weight = nullptr, const Tensor* bias = nullptr,
                         float eps = 1e-5f) {
    auto shape = input.shape();
    int normalized_size = 1;
    for (int s : normalized_shape) normalized_size *= s;
    
    int batch_size = input.numel() / normalized_size;
    std::vector<float> result(input.numel());
    
    for (int b = 0; b < batch_size; ++b) {
        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < normalized_size; ++i) {
            mean += input.data()[b * normalized_size + i];
        }
        mean /= normalized_size;
        
        // Compute variance
        float var = 0.0f;
        for (int i = 0; i < normalized_size; ++i) {
            float diff = input.data()[b * normalized_size + i] - mean;
            var += diff * diff;
        }
        var /= normalized_size;
        
        // Normalize
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < normalized_size; ++i) {
            float normalized = (input.data()[b * normalized_size + i] - mean) * inv_std;
            
            if (weight != nullptr) {
                normalized *= weight->data()[i];
            }
            if (bias != nullptr) {
                normalized += bias->data()[i];
            }
            
            result[b * normalized_size + i] = normalized;
        }
    }
    
    return Tensor(result, shape);
}

// ============================================================================
// Pooling
// ============================================================================

inline Tensor max_pool1d(const Tensor& input, int kernel_size, int stride = -1, int padding = 0) {
    if (stride < 0) stride = kernel_size;
    
    auto shape = input.shape();
    int length = shape.back();
    int out_length = (length + 2 * padding - kernel_size) / stride + 1;
    
    std::vector<float> result(out_length);
    
    for (int i = 0; i < out_length; ++i) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < kernel_size; ++k) {
            int idx = i * stride + k - padding;
            if (idx >= 0 && idx < length) {
                max_val = std::max(max_val, input.data()[idx]);
            }
        }
        result[i] = max_val;
    }
    
    return Tensor(result, {out_length});
}

inline Tensor avg_pool1d(const Tensor& input, int kernel_size, int stride = -1, int padding = 0) {
    if (stride < 0) stride = kernel_size;
    
    auto shape = input.shape();
    int length = shape.back();
    int out_length = (length + 2 * padding - kernel_size) / stride + 1;
    
    std::vector<float> result(out_length);
    
    for (int i = 0; i < out_length; ++i) {
        float sum = 0.0f;
        int count = 0;
        for (int k = 0; k < kernel_size; ++k) {
            int idx = i * stride + k - padding;
            if (idx >= 0 && idx < length) {
                sum += input.data()[idx];
                count++;
            }
        }
        result[i] = sum / count;
    }
    
    return Tensor(result, {out_length});
}

inline Tensor adaptive_avg_pool1d(const Tensor& input, int output_size) {
    auto shape = input.shape();
    int input_size = shape.back();
    
    std::vector<float> result(output_size);
    
    for (int i = 0; i < output_size; ++i) {
        int start = i * input_size / output_size;
        int end = (i + 1) * input_size / output_size;
        
        float sum = 0.0f;
        for (int j = start; j < end; ++j) {
            sum += input.data()[j];
        }
        result[i] = sum / (end - start);
    }
    
    return Tensor(result, {output_size});
}

// ============================================================================
// Linear
// ============================================================================

inline Tensor linear(const Tensor& input, const Tensor& weight, const Tensor* bias = nullptr) {
    Tensor output = input.matmul(weight.transpose(-2, -1));
    
    if (bias != nullptr) {
        output = output + *bias;
    }
    
    return output;
}

inline Tensor bilinear(const Tensor& input1, const Tensor& input2, 
                       const Tensor& weight, const Tensor* bias = nullptr) {
    // Simplified bilinear: out_i = x1^T W_i x2 + b_i
    auto w_shape = weight.shape();
    int out_features = w_shape[0];
    int in1_features = w_shape[1];
    int in2_features = w_shape[2];
    
    std::vector<float> result(out_features);
    
    for (int o = 0; o < out_features; ++o) {
        float val = 0.0f;
        for (int i = 0; i < in1_features; ++i) {
            for (int j = 0; j < in2_features; ++j) {
                val += input1.data()[i] * weight.data()[o * in1_features * in2_features + i * in2_features + j] * input2.data()[j];
            }
        }
        result[o] = val;
    }
    
    Tensor output(result, {out_features});
    
    if (bias != nullptr) {
        output = output + *bias;
    }
    
    return output;
}

// ============================================================================
// Padding
// ============================================================================

inline Tensor pad(const Tensor& input, const std::vector<int>& padding, 
                  const std::string& mode = "constant", float value = 0.0f) {
    auto shape = input.shape();
    int ndim = static_cast<int>(shape.size());
    
    // Padding format: (left, right) or (left, right, top, bottom) etc.
    std::vector<int> new_shape = shape;
    
    for (size_t i = 0; i < padding.size() / 2; ++i) {
        int dim = ndim - 1 - static_cast<int>(i);
        new_shape[dim] += padding[2 * i] + padding[2 * i + 1];
    }
    
    int total_size = 1;
    for (int s : new_shape) total_size *= s;
    
    std::vector<float> result(total_size, value);
    
    // Copy input to padded tensor (simplified for 1D case)
    if (ndim == 1 && padding.size() >= 2) {
        int left_pad = padding[0];
        for (int i = 0; i < shape[0]; ++i) {
            result[left_pad + i] = input.data()[i];
        }
    }
    
    return Tensor(result, new_shape);
}

inline Tensor reflection_pad1d(const Tensor& input, int padding) {
    auto shape = input.shape();
    int length = shape.back();
    int new_length = length + 2 * padding;
    
    std::vector<float> result(new_length);
    
    for (int i = 0; i < new_length; ++i) {
        int idx;
        if (i < padding) {
            idx = padding - i;
        } else if (i >= length + padding) {
            idx = 2 * length + padding - i - 2;
        } else {
            idx = i - padding;
        }
        result[i] = input.data()[idx];
    }
    
    return Tensor(result, {new_length});
}

inline Tensor replicate_pad1d(const Tensor& input, int padding) {
    auto shape = input.shape();
    int length = shape.back();
    int new_length = length + 2 * padding;
    
    std::vector<float> result(new_length);
    
    for (int i = 0; i < new_length; ++i) {
        int idx = std::max(0, std::min(length - 1, i - padding));
        result[i] = input.data()[idx];
    }
    
    return Tensor(result, {new_length});
}

// ============================================================================
// Interpolation
// ============================================================================

inline Tensor interpolate(const Tensor& input, int size = -1, float scale_factor = -1.0f,
                          const std::string& mode = "nearest") {
    auto shape = input.shape();
    int in_size = shape.back();
    int out_size;
    
    if (size > 0) {
        out_size = size;
    } else if (scale_factor > 0.0f) {
        out_size = static_cast<int>(in_size * scale_factor);
    } else {
        return input;
    }
    
    std::vector<float> result(out_size);
    
    if (mode == "nearest") {
        for (int i = 0; i < out_size; ++i) {
            int src_idx = static_cast<int>(static_cast<float>(i) * in_size / out_size);
            src_idx = std::min(src_idx, in_size - 1);
            result[i] = input.data()[src_idx];
        }
    } else if (mode == "linear") {
        for (int i = 0; i < out_size; ++i) {
            float src_pos = static_cast<float>(i) * (in_size - 1) / (out_size - 1);
            int left = static_cast<int>(src_pos);
            int right = std::min(left + 1, in_size - 1);
            float t = src_pos - left;
            result[i] = input.data()[left] * (1.0f - t) + input.data()[right] * t;
        }
    }
    
    return Tensor(result, {out_size});
}

// ============================================================================
// Distance Functions
// ============================================================================

inline Tensor pairwise_distance(const Tensor& x1, const Tensor& x2, float p = 2.0f,
                                 float eps = 1e-6f, bool keepdim = false) {
    Tensor diff = x1 - x2;
    
    std::vector<float> result(diff.numel());
    for (size_t i = 0; i < diff.numel(); ++i) {
        result[i] = std::pow(std::abs(diff.data()[i]), p);
    }
    
    Tensor powered(result, diff.shape());
    Tensor summed = powered.sum();
    
    std::vector<float> final_result = {std::pow(summed.data()[0] + eps, 1.0f / p)};
    
    if (keepdim) {
        return Tensor(final_result, {1});
    }
    return Tensor(final_result, {});
}

inline Tensor cosine_similarity(const Tensor& x1, const Tensor& x2, int dim = 1, float eps = 1e-8f) {
    // Dot product
    float dot = 0.0f;
    for (size_t i = 0; i < x1.numel(); ++i) {
        dot += x1.data()[i] * x2.data()[i];
    }
    
    // Norms
    float norm1 = 0.0f, norm2 = 0.0f;
    for (size_t i = 0; i < x1.numel(); ++i) {
        norm1 += x1.data()[i] * x1.data()[i];
        norm2 += x2.data()[i] * x2.data()[i];
    }
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    float similarity = dot / std::max(norm1 * norm2, eps);
    
    return Tensor({similarity}, {});
}

// ============================================================================
// One-hot encoding
// ============================================================================

inline Tensor one_hot(const Tensor& indices, int num_classes) {
    int batch_size = static_cast<int>(indices.numel());
    std::vector<float> result(batch_size * num_classes, 0.0f);
    
    for (int i = 0; i < batch_size; ++i) {
        int idx = static_cast<int>(indices.data()[i]);
        if (idx >= 0 && idx < num_classes) {
            result[i * num_classes + idx] = 1.0f;
        }
    }
    
    return Tensor(result, {batch_size, num_classes});
}

} // namespace functional
} // namespace nn
} // namespace neurova
