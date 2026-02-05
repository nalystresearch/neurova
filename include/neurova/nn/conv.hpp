// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file conv.hpp
 * @brief Convolutional layers
 * 
 * Neurova implementation of 1D, 2D, and 3D convolutions.
 */

#pragma once

#include "layers.hpp"
#include <cmath>
#include <tuple>

namespace neurova {
namespace nn {

/**
 * @brief 1D Convolution layer
 */
class Conv1d : public Module {
private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    size_t dilation_;
    size_t groups_;
    bool use_bias_;
    Parameter weight_;
    Parameter bias_;
    
public:
    Conv1d(size_t in_channels, size_t out_channels, size_t kernel_size,
           size_t stride = 1, size_t padding = 0, size_t dilation = 1,
           size_t groups = 1, bool bias = true)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_(kernel_size), stride_(stride), padding_(padding),
          dilation_(dilation), groups_(groups), use_bias_(bias) {
        
        float k = 1.0f / std::sqrt(static_cast<float>(in_channels * kernel_size));
        
        // Weight shape: (out_channels, in_channels/groups, kernel_size)
        size_t weight_size = out_channels * (in_channels / groups) * kernel_size;
        std::vector<float> weight_data(weight_size);
        for (auto& w : weight_data) {
            w = -k + 2.0f * k * static_cast<float>(rand()) / RAND_MAX;
        }
        weight_ = Parameter(Tensor::create(weight_data,
            {out_channels, in_channels / groups, kernel_size}, true));
        register_parameter("weight", weight_);
        
        if (bias) {
            std::vector<float> bias_data(out_channels);
            for (auto& b : bias_data) {
                b = -k + 2.0f * k * static_cast<float>(rand()) / RAND_MAX;
            }
            bias_ = Parameter(Tensor::create(bias_data, {out_channels}, true));
            register_parameter("bias", bias_);
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        // x: (batch, in_channels, length)
        size_t batch = x->shape[0];
        size_t in_len = x->shape[2];
        size_t out_len = (in_len + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
        
        std::vector<float> output(batch * out_channels_ * out_len, 0.0f);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t oc = 0; oc < out_channels_; ++oc) {
                for (size_t ol = 0; ol < out_len; ++ol) {
                    float sum = 0.0f;
                    for (size_t ic = 0; ic < in_channels_; ++ic) {
                        for (size_t k = 0; k < kernel_size_; ++k) {
                            int il = static_cast<int>(ol * stride_ + k * dilation_) - static_cast<int>(padding_);
                            if (il >= 0 && il < static_cast<int>(in_len)) {
                                size_t x_idx = b * in_channels_ * in_len + ic * in_len + il;
                                size_t w_idx = oc * (in_channels_ / groups_) * kernel_size_ + 
                                              ic * kernel_size_ + k;
                                sum += x->data[x_idx] * weight_.data->data[w_idx];
                            }
                        }
                    }
                    if (use_bias_) sum += bias_.data->data[oc];
                    output[b * out_channels_ * out_len + oc * out_len + ol] = sum;
                }
            }
        }
        
        return std::make_shared<Tensor>(output, {batch, out_channels_, out_len}, x->requires_grad);
    }
};

/**
 * @brief 2D Convolution layer
 */
class Conv2d : public Module {
private:
    size_t in_channels_;
    size_t out_channels_;
    std::pair<size_t, size_t> kernel_size_;
    std::pair<size_t, size_t> stride_;
    std::pair<size_t, size_t> padding_;
    std::pair<size_t, size_t> dilation_;
    size_t groups_;
    bool use_bias_;
    Parameter weight_;
    Parameter bias_;
    
public:
    Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
           size_t stride = 1, size_t padding = 0, size_t dilation = 1,
           size_t groups = 1, bool bias = true)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_({kernel_size, kernel_size}),
          stride_({stride, stride}), padding_({padding, padding}),
          dilation_({dilation, dilation}), groups_(groups), use_bias_(bias) {
        initWeights();
    }
    
    Conv2d(size_t in_channels, size_t out_channels, 
           std::pair<size_t, size_t> kernel_size,
           std::pair<size_t, size_t> stride = {1, 1},
           std::pair<size_t, size_t> padding = {0, 0},
           std::pair<size_t, size_t> dilation = {1, 1},
           size_t groups = 1, bool bias = true)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_(kernel_size), stride_(stride), padding_(padding),
          dilation_(dilation), groups_(groups), use_bias_(bias) {
        initWeights();
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        // x: (batch, in_channels, height, width)
        size_t batch = x->shape[0];
        size_t in_h = x->shape[2];
        size_t in_w = x->shape[3];
        
        size_t out_h = (in_h + 2 * padding_.first - dilation_.first * (kernel_size_.first - 1) - 1) / stride_.first + 1;
        size_t out_w = (in_w + 2 * padding_.second - dilation_.second * (kernel_size_.second - 1) - 1) / stride_.second + 1;
        
        std::vector<float> output(batch * out_channels_ * out_h * out_w, 0.0f);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t oc = 0; oc < out_channels_; ++oc) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        float sum = 0.0f;
                        
                        for (size_t ic = 0; ic < in_channels_ / groups_; ++ic) {
                            for (size_t kh = 0; kh < kernel_size_.first; ++kh) {
                                for (size_t kw = 0; kw < kernel_size_.second; ++kw) {
                                    int ih = static_cast<int>(oh * stride_.first + kh * dilation_.first) - 
                                             static_cast<int>(padding_.first);
                                    int iw = static_cast<int>(ow * stride_.second + kw * dilation_.second) - 
                                             static_cast<int>(padding_.second);
                                    
                                    if (ih >= 0 && ih < static_cast<int>(in_h) &&
                                        iw >= 0 && iw < static_cast<int>(in_w)) {
                                        size_t x_idx = b * in_channels_ * in_h * in_w + 
                                                      ic * in_h * in_w + ih * in_w + iw;
                                        size_t w_idx = oc * (in_channels_ / groups_) * kernel_size_.first * kernel_size_.second +
                                                      ic * kernel_size_.first * kernel_size_.second +
                                                      kh * kernel_size_.second + kw;
                                        sum += x->data[x_idx] * weight_.data->data[w_idx];
                                    }
                                }
                            }
                        }
                        
                        if (use_bias_) sum += bias_.data->data[oc];
                        output[b * out_channels_ * out_h * out_w + oc * out_h * out_w + oh * out_w + ow] = sum;
                    }
                }
            }
        }
        
        return std::make_shared<Tensor>(output, {batch, out_channels_, out_h, out_w}, x->requires_grad);
    }
    
private:
    void initWeights() {
        float k = 1.0f / std::sqrt(static_cast<float>(in_channels_ * kernel_size_.first * kernel_size_.second));
        
        size_t weight_size = out_channels_ * (in_channels_ / groups_) * kernel_size_.first * kernel_size_.second;
        std::vector<float> weight_data(weight_size);
        for (auto& w : weight_data) {
            w = -k + 2.0f * k * static_cast<float>(rand()) / RAND_MAX;
        }
        weight_ = Parameter(Tensor::create(weight_data,
            {out_channels_, in_channels_ / groups_, kernel_size_.first, kernel_size_.second}, true));
        register_parameter("weight", weight_);
        
        if (use_bias_) {
            std::vector<float> bias_data(out_channels_);
            for (auto& b : bias_data) {
                b = -k + 2.0f * k * static_cast<float>(rand()) / RAND_MAX;
            }
            bias_ = Parameter(Tensor::create(bias_data, {out_channels_}, true));
            register_parameter("bias", bias_);
        }
    }
};

/**
 * @brief 3D Convolution layer
 */
class Conv3d : public Module {
private:
    size_t in_channels_;
    size_t out_channels_;
    std::tuple<size_t, size_t, size_t> kernel_size_;
    std::tuple<size_t, size_t, size_t> stride_;
    std::tuple<size_t, size_t, size_t> padding_;
    bool use_bias_;
    Parameter weight_;
    Parameter bias_;
    
public:
    Conv3d(size_t in_channels, size_t out_channels, size_t kernel_size,
           size_t stride = 1, size_t padding = 0, bool bias = true)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_({kernel_size, kernel_size, kernel_size}),
          stride_({stride, stride, stride}),
          padding_({padding, padding, padding}), use_bias_(bias) {
        
        float k = 1.0f / std::sqrt(static_cast<float>(in_channels * kernel_size * kernel_size * kernel_size));
        
        size_t weight_size = out_channels * in_channels * kernel_size * kernel_size * kernel_size;
        std::vector<float> weight_data(weight_size);
        for (auto& w : weight_data) {
            w = -k + 2.0f * k * static_cast<float>(rand()) / RAND_MAX;
        }
        weight_ = Parameter(Tensor::create(weight_data,
            {out_channels, in_channels, kernel_size, kernel_size, kernel_size}, true));
        register_parameter("weight", weight_);
        
        if (bias) {
            std::vector<float> bias_data(out_channels);
            for (auto& b : bias_data) {
                b = -k + 2.0f * k * static_cast<float>(rand()) / RAND_MAX;
            }
            bias_ = Parameter(Tensor::create(bias_data, {out_channels}, true));
            register_parameter("bias", bias_);
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        // Simplified 3D conv - full implementation would mirror 2D
        return x;  // Placeholder
    }
};

/**
 * @brief 2D Transposed Convolution (Deconvolution)
 */
class ConvTranspose2d : public Module {
private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    size_t output_padding_;
    bool use_bias_;
    Parameter weight_;
    Parameter bias_;
    
public:
    ConvTranspose2d(size_t in_channels, size_t out_channels, size_t kernel_size,
                    size_t stride = 1, size_t padding = 0, size_t output_padding = 0,
                    bool bias = true)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_(kernel_size), stride_(stride), padding_(padding),
          output_padding_(output_padding), use_bias_(bias) {
        
        float k = 1.0f / std::sqrt(static_cast<float>(in_channels * kernel_size * kernel_size));
        
        size_t weight_size = in_channels * out_channels * kernel_size * kernel_size;
        std::vector<float> weight_data(weight_size);
        for (auto& w : weight_data) {
            w = -k + 2.0f * k * static_cast<float>(rand()) / RAND_MAX;
        }
        weight_ = Parameter(Tensor::create(weight_data,
            {in_channels, out_channels, kernel_size, kernel_size}, true));
        register_parameter("weight", weight_);
        
        if (bias) {
            std::vector<float> bias_data(out_channels);
            for (auto& b : bias_data) {
                b = -k + 2.0f * k * static_cast<float>(rand()) / RAND_MAX;
            }
            bias_ = Parameter(Tensor::create(bias_data, {out_channels}, true));
            register_parameter("bias", bias_);
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        // x: (batch, in_channels, height, width)
        size_t batch = x->shape[0];
        size_t in_h = x->shape[2];
        size_t in_w = x->shape[3];
        
        size_t out_h = (in_h - 1) * stride_ - 2 * padding_ + kernel_size_ + output_padding_;
        size_t out_w = (in_w - 1) * stride_ - 2 * padding_ + kernel_size_ + output_padding_;
        
        std::vector<float> output(batch * out_channels_ * out_h * out_w, 0.0f);
        
        // Transposed convolution implementation
        for (size_t b = 0; b < batch; ++b) {
            for (size_t ic = 0; ic < in_channels_; ++ic) {
                for (size_t ih = 0; ih < in_h; ++ih) {
                    for (size_t iw = 0; iw < in_w; ++iw) {
                        float val = x->data[b * in_channels_ * in_h * in_w + ic * in_h * in_w + ih * in_w + iw];
                        
                        for (size_t oc = 0; oc < out_channels_; ++oc) {
                            for (size_t kh = 0; kh < kernel_size_; ++kh) {
                                for (size_t kw = 0; kw < kernel_size_; ++kw) {
                                    size_t oh = ih * stride_ + kh - padding_;
                                    size_t ow = iw * stride_ + kw - padding_;
                                    
                                    if (oh < out_h && ow < out_w) {
                                        size_t w_idx = ic * out_channels_ * kernel_size_ * kernel_size_ +
                                                      oc * kernel_size_ * kernel_size_ +
                                                      kh * kernel_size_ + kw;
                                        output[b * out_channels_ * out_h * out_w + 
                                               oc * out_h * out_w + oh * out_w + ow] += 
                                            val * weight_.data->data[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            if (use_bias_) {
                for (size_t oc = 0; oc < out_channels_; ++oc) {
                    for (size_t oh = 0; oh < out_h; ++oh) {
                        for (size_t ow = 0; ow < out_w; ++ow) {
                            output[b * out_channels_ * out_h * out_w + 
                                   oc * out_h * out_w + oh * out_w + ow] += bias_.data->data[oc];
                        }
                    }
                }
            }
        }
        
        return std::make_shared<Tensor>(output, {batch, out_channels_, out_h, out_w}, x->requires_grad);
    }
};

// Aliases
using ConvTranspose1d = Conv1d;  // Placeholder
using ConvTranspose3d = Conv3d;  // Placeholder

} // namespace nn
} // namespace neurova
