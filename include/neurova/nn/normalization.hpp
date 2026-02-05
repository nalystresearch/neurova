// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file normalization.hpp
 * @brief Normalization layers
 * 
 * Neurova implementation of batch, layer, group, and instance normalization.
 */

#pragma once

#include "layers.hpp"
#include <cmath>

namespace neurova {
namespace nn {

/**
 * @brief 1D Batch Normalization
 */
class BatchNorm1d : public Module {
private:
    size_t num_features_;
    float eps_;
    float momentum_;
    bool affine_;
    bool track_running_stats_;
    
    Parameter weight_;
    Parameter bias_;
    TensorPtr running_mean_;
    TensorPtr running_var_;
    size_t num_batches_tracked_ = 0;
    
public:
    BatchNorm1d(size_t num_features, float eps = 1e-5f, float momentum = 0.1f,
                bool affine = true, bool track_running_stats = true)
        : num_features_(num_features), eps_(eps), momentum_(momentum),
          affine_(affine), track_running_stats_(track_running_stats) {
        
        if (affine) {
            weight_ = Parameter(Tensor::ones({num_features}, true));
            bias_ = Parameter(Tensor::zeros({num_features}, true));
            register_parameter("weight", weight_);
            register_parameter("bias", bias_);
        }
        
        if (track_running_stats) {
            running_mean_ = Tensor::zeros({num_features}, false);
            running_var_ = Tensor::ones({num_features}, false);
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        // x: (batch, channels) or (batch, channels, length)
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t length = x->shape.size() > 2 ? x->shape[2] : 1;
        
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        
        for (size_t c = 0; c < channels; ++c) {
            float mean, var;
            
            if (training_ || !track_running_stats_) {
                // Compute batch statistics
                mean = 0.0f;
                for (size_t b = 0; b < batch; ++b) {
                    for (size_t l = 0; l < length; ++l) {
                        mean += x->data[b * channels * length + c * length + l];
                    }
                }
                mean /= batch * length;
                
                var = 0.0f;
                for (size_t b = 0; b < batch; ++b) {
                    for (size_t l = 0; l < length; ++l) {
                        float diff = x->data[b * channels * length + c * length + l] - mean;
                        var += diff * diff;
                    }
                }
                var /= batch * length;
                
                if (track_running_stats_ && training_) {
                    running_mean_->data[c] = (1 - momentum_) * running_mean_->data[c] + momentum_ * mean;
                    running_var_->data[c] = (1 - momentum_) * running_var_->data[c] + momentum_ * var;
                    num_batches_tracked_++;
                }
            } else {
                mean = running_mean_->data[c];
                var = running_var_->data[c];
            }
            
            float std = std::sqrt(var + eps_);
            
            for (size_t b = 0; b < batch; ++b) {
                for (size_t l = 0; l < length; ++l) {
                    size_t idx = b * channels * length + c * length + l;
                    float normalized = (out->data[idx] - mean) / std;
                    if (affine_) {
                        normalized = normalized * weight_.data->data[c] + bias_.data->data[c];
                    }
                    out->data[idx] = normalized;
                }
            }
        }
        
        return out;
    }
};

/**
 * @brief 2D Batch Normalization
 */
class BatchNorm2d : public Module {
private:
    size_t num_features_;
    float eps_;
    float momentum_;
    bool affine_;
    bool track_running_stats_;
    
    Parameter weight_;
    Parameter bias_;
    TensorPtr running_mean_;
    TensorPtr running_var_;
    
public:
    BatchNorm2d(size_t num_features, float eps = 1e-5f, float momentum = 0.1f,
                bool affine = true, bool track_running_stats = true)
        : num_features_(num_features), eps_(eps), momentum_(momentum),
          affine_(affine), track_running_stats_(track_running_stats) {
        
        if (affine) {
            weight_ = Parameter(Tensor::ones({num_features}, true));
            bias_ = Parameter(Tensor::zeros({num_features}, true));
            register_parameter("weight", weight_);
            register_parameter("bias", bias_);
        }
        
        if (track_running_stats) {
            running_mean_ = Tensor::zeros({num_features}, false);
            running_var_ = Tensor::ones({num_features}, false);
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        // x: (batch, channels, height, width)
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t height = x->shape[2];
        size_t width = x->shape[3];
        size_t spatial = height * width;
        
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        
        for (size_t c = 0; c < channels; ++c) {
            float mean, var;
            
            if (training_ || !track_running_stats_) {
                mean = 0.0f;
                for (size_t b = 0; b < batch; ++b) {
                    for (size_t s = 0; s < spatial; ++s) {
                        mean += x->data[b * channels * spatial + c * spatial + s];
                    }
                }
                mean /= batch * spatial;
                
                var = 0.0f;
                for (size_t b = 0; b < batch; ++b) {
                    for (size_t s = 0; s < spatial; ++s) {
                        float diff = x->data[b * channels * spatial + c * spatial + s] - mean;
                        var += diff * diff;
                    }
                }
                var /= batch * spatial;
                
                if (track_running_stats_ && training_) {
                    running_mean_->data[c] = (1 - momentum_) * running_mean_->data[c] + momentum_ * mean;
                    running_var_->data[c] = (1 - momentum_) * running_var_->data[c] + momentum_ * var;
                }
            } else {
                mean = running_mean_->data[c];
                var = running_var_->data[c];
            }
            
            float std = std::sqrt(var + eps_);
            
            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < spatial; ++s) {
                    size_t idx = b * channels * spatial + c * spatial + s;
                    float normalized = (out->data[idx] - mean) / std;
                    if (affine_) {
                        normalized = normalized * weight_.data->data[c] + bias_.data->data[c];
                    }
                    out->data[idx] = normalized;
                }
            }
        }
        
        return out;
    }
};

/**
 * @brief 3D Batch Normalization
 */
class BatchNorm3d : public BatchNorm2d {
public:
    using BatchNorm2d::BatchNorm2d;
};

/**
 * @brief Layer Normalization
 */
class LayerNorm : public Module {
private:
    std::vector<size_t> normalized_shape_;
    float eps_;
    bool elementwise_affine_;
    Parameter weight_;
    Parameter bias_;
    
public:
    LayerNorm(const std::vector<size_t>& normalized_shape, float eps = 1e-5f,
              bool elementwise_affine = true)
        : normalized_shape_(normalized_shape), eps_(eps), 
          elementwise_affine_(elementwise_affine) {
        
        size_t num_elements = 1;
        for (auto s : normalized_shape) num_elements *= s;
        
        if (elementwise_affine) {
            weight_ = Parameter(Tensor::ones(normalized_shape, true));
            bias_ = Parameter(Tensor::zeros(normalized_shape, true));
            register_parameter("weight", weight_);
            register_parameter("bias", bias_);
        }
    }
    
    LayerNorm(size_t normalized_shape, float eps = 1e-5f, bool elementwise_affine = true)
        : LayerNorm(std::vector<size_t>{normalized_shape}, eps, elementwise_affine) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        size_t norm_size = 1;
        for (auto s : normalized_shape_) norm_size *= s;
        
        size_t batch_size = x->size() / norm_size;
        
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        
        for (size_t b = 0; b < batch_size; ++b) {
            // Compute mean
            float mean = 0.0f;
            for (size_t i = 0; i < norm_size; ++i) {
                mean += x->data[b * norm_size + i];
            }
            mean /= norm_size;
            
            // Compute variance
            float var = 0.0f;
            for (size_t i = 0; i < norm_size; ++i) {
                float diff = x->data[b * norm_size + i] - mean;
                var += diff * diff;
            }
            var /= norm_size;
            
            float std = std::sqrt(var + eps_);
            
            // Normalize and apply affine
            for (size_t i = 0; i < norm_size; ++i) {
                float normalized = (out->data[b * norm_size + i] - mean) / std;
                if (elementwise_affine_) {
                    normalized = normalized * weight_.data->data[i] + bias_.data->data[i];
                }
                out->data[b * norm_size + i] = normalized;
            }
        }
        
        return out;
    }
};

/**
 * @brief Group Normalization
 */
class GroupNorm : public Module {
private:
    size_t num_groups_;
    size_t num_channels_;
    float eps_;
    bool affine_;
    Parameter weight_;
    Parameter bias_;
    
public:
    GroupNorm(size_t num_groups, size_t num_channels, float eps = 1e-5f, bool affine = true)
        : num_groups_(num_groups), num_channels_(num_channels), eps_(eps), affine_(affine) {
        
        if (affine) {
            weight_ = Parameter(Tensor::ones({num_channels}, true));
            bias_ = Parameter(Tensor::zeros({num_channels}, true));
            register_parameter("weight", weight_);
            register_parameter("bias", bias_);
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        // x: (batch, channels, ...)
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t spatial = x->size() / (batch * channels);
        size_t channels_per_group = channels / num_groups_;
        
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t g = 0; g < num_groups_; ++g) {
                // Compute mean for group
                float mean = 0.0f;
                for (size_t c = g * channels_per_group; c < (g + 1) * channels_per_group; ++c) {
                    for (size_t s = 0; s < spatial; ++s) {
                        mean += x->data[b * channels * spatial + c * spatial + s];
                    }
                }
                mean /= channels_per_group * spatial;
                
                // Compute variance
                float var = 0.0f;
                for (size_t c = g * channels_per_group; c < (g + 1) * channels_per_group; ++c) {
                    for (size_t s = 0; s < spatial; ++s) {
                        float diff = x->data[b * channels * spatial + c * spatial + s] - mean;
                        var += diff * diff;
                    }
                }
                var /= channels_per_group * spatial;
                
                float std = std::sqrt(var + eps_);
                
                // Normalize
                for (size_t c = g * channels_per_group; c < (g + 1) * channels_per_group; ++c) {
                    for (size_t s = 0; s < spatial; ++s) {
                        size_t idx = b * channels * spatial + c * spatial + s;
                        float normalized = (out->data[idx] - mean) / std;
                        if (affine_) {
                            normalized = normalized * weight_.data->data[c] + bias_.data->data[c];
                        }
                        out->data[idx] = normalized;
                    }
                }
            }
        }
        
        return out;
    }
};

/**
 * @brief 1D Instance Normalization
 */
class InstanceNorm1d : public Module {
private:
    size_t num_features_;
    float eps_;
    float momentum_;
    bool affine_;
    Parameter weight_;
    Parameter bias_;
    
public:
    InstanceNorm1d(size_t num_features, float eps = 1e-5f, float momentum = 0.1f, 
                   bool affine = false)
        : num_features_(num_features), eps_(eps), momentum_(momentum), affine_(affine) {
        
        if (affine) {
            weight_ = Parameter(Tensor::ones({num_features}, true));
            bias_ = Parameter(Tensor::zeros({num_features}, true));
            register_parameter("weight", weight_);
            register_parameter("bias", bias_);
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        // x: (batch, channels, length)
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t length = x->shape[2];
        
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                // Compute mean for this instance
                float mean = 0.0f;
                for (size_t l = 0; l < length; ++l) {
                    mean += x->data[b * channels * length + c * length + l];
                }
                mean /= length;
                
                // Compute variance
                float var = 0.0f;
                for (size_t l = 0; l < length; ++l) {
                    float diff = x->data[b * channels * length + c * length + l] - mean;
                    var += diff * diff;
                }
                var /= length;
                
                float std = std::sqrt(var + eps_);
                
                // Normalize
                for (size_t l = 0; l < length; ++l) {
                    size_t idx = b * channels * length + c * length + l;
                    float normalized = (out->data[idx] - mean) / std;
                    if (affine_) {
                        normalized = normalized * weight_.data->data[c] + bias_.data->data[c];
                    }
                    out->data[idx] = normalized;
                }
            }
        }
        
        return out;
    }
};

/**
 * @brief 2D Instance Normalization
 */
class InstanceNorm2d : public Module {
private:
    size_t num_features_;
    float eps_;
    bool affine_;
    Parameter weight_;
    Parameter bias_;
    
public:
    InstanceNorm2d(size_t num_features, float eps = 1e-5f, bool affine = false)
        : num_features_(num_features), eps_(eps), affine_(affine) {
        
        if (affine) {
            weight_ = Parameter(Tensor::ones({num_features}, true));
            bias_ = Parameter(Tensor::zeros({num_features}, true));
            register_parameter("weight", weight_);
            register_parameter("bias", bias_);
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        // x: (batch, channels, height, width)
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t spatial = x->shape[2] * x->shape[3];
        
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                // Compute mean
                float mean = 0.0f;
                for (size_t s = 0; s < spatial; ++s) {
                    mean += x->data[b * channels * spatial + c * spatial + s];
                }
                mean /= spatial;
                
                // Compute variance
                float var = 0.0f;
                for (size_t s = 0; s < spatial; ++s) {
                    float diff = x->data[b * channels * spatial + c * spatial + s] - mean;
                    var += diff * diff;
                }
                var /= spatial;
                
                float std = std::sqrt(var + eps_);
                
                // Normalize
                for (size_t s = 0; s < spatial; ++s) {
                    size_t idx = b * channels * spatial + c * spatial + s;
                    float normalized = (out->data[idx] - mean) / std;
                    if (affine_) {
                        normalized = normalized * weight_.data->data[c] + bias_.data->data[c];
                    }
                    out->data[idx] = normalized;
                }
            }
        }
        
        return out;
    }
};

/**
 * @brief RMS Normalization
 */
class RMSNorm : public Module {
private:
    size_t normalized_shape_;
    float eps_;
    Parameter weight_;
    
public:
    RMSNorm(size_t normalized_shape, float eps = 1e-6f)
        : normalized_shape_(normalized_shape), eps_(eps) {
        weight_ = Parameter(Tensor::ones({normalized_shape}, true));
        register_parameter("weight", weight_);
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        size_t batch_size = x->size() / normalized_shape_;
        
        auto out = std::make_shared<Tensor>(x->data, x->shape, x->requires_grad);
        
        for (size_t b = 0; b < batch_size; ++b) {
            // Compute RMS
            float rms = 0.0f;
            for (size_t i = 0; i < normalized_shape_; ++i) {
                float val = x->data[b * normalized_shape_ + i];
                rms += val * val;
            }
            rms = std::sqrt(rms / normalized_shape_ + eps_);
            
            // Normalize and scale
            for (size_t i = 0; i < normalized_shape_; ++i) {
                out->data[b * normalized_shape_ + i] = 
                    (out->data[b * normalized_shape_ + i] / rms) * weight_.data->data[i];
            }
        }
        
        return out;
    }
};

} // namespace nn
} // namespace neurova
