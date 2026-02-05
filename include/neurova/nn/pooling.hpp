// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file pooling.hpp
 * @brief Pooling layers
 * 
 * Neurova implementation of pooling operations.
 */

#pragma once

#include "layers.hpp"
#include <algorithm>
#include <limits>

namespace neurova {
namespace nn {

/**
 * @brief 1D Max Pooling
 */
class MaxPool1d : public Module {
private:
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    bool ceil_mode_;
    bool return_indices_;
    
public:
    MaxPool1d(size_t kernel_size, size_t stride = 0, size_t padding = 0,
              bool ceil_mode = false, bool return_indices = false)
        : kernel_size_(kernel_size), stride_(stride == 0 ? kernel_size : stride),
          padding_(padding), ceil_mode_(ceil_mode), return_indices_(return_indices) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        // x: (batch, channels, length)
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t in_len = x->shape[2];
        
        size_t out_len;
        if (ceil_mode_) {
            out_len = (in_len + 2 * padding_ - kernel_size_ + stride_ - 1) / stride_ + 1;
        } else {
            out_len = (in_len + 2 * padding_ - kernel_size_) / stride_ + 1;
        }
        
        std::vector<float> output(batch * channels * out_len);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t o = 0; o < out_len; ++o) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    
                    for (size_t k = 0; k < kernel_size_; ++k) {
                        int idx = static_cast<int>(o * stride_ + k) - static_cast<int>(padding_);
                        if (idx >= 0 && idx < static_cast<int>(in_len)) {
                            float val = x->data[b * channels * in_len + c * in_len + idx];
                            max_val = std::max(max_val, val);
                        }
                    }
                    
                    output[b * channels * out_len + c * out_len + o] = max_val;
                }
            }
        }
        
        return std::make_shared<Tensor>(output, {batch, channels, out_len}, x->requires_grad);
    }
};

/**
 * @brief 2D Max Pooling
 */
class MaxPool2d : public Module {
private:
    std::pair<size_t, size_t> kernel_size_;
    std::pair<size_t, size_t> stride_;
    std::pair<size_t, size_t> padding_;
    bool ceil_mode_;
    
public:
    MaxPool2d(size_t kernel_size, size_t stride = 0, size_t padding = 0, bool ceil_mode = false)
        : kernel_size_({kernel_size, kernel_size}),
          stride_({stride == 0 ? kernel_size : stride, stride == 0 ? kernel_size : stride}),
          padding_({padding, padding}), ceil_mode_(ceil_mode) {}
    
    MaxPool2d(std::pair<size_t, size_t> kernel_size, 
              std::pair<size_t, size_t> stride = {0, 0},
              std::pair<size_t, size_t> padding = {0, 0}, bool ceil_mode = false)
        : kernel_size_(kernel_size),
          stride_({stride.first == 0 ? kernel_size.first : stride.first,
                   stride.second == 0 ? kernel_size.second : stride.second}),
          padding_(padding), ceil_mode_(ceil_mode) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        // x: (batch, channels, height, width)
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t in_h = x->shape[2];
        size_t in_w = x->shape[3];
        
        size_t out_h = (in_h + 2 * padding_.first - kernel_size_.first) / stride_.first + 1;
        size_t out_w = (in_w + 2 * padding_.second - kernel_size_.second) / stride_.second + 1;
        
        std::vector<float> output(batch * channels * out_h * out_w);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        
                        for (size_t kh = 0; kh < kernel_size_.first; ++kh) {
                            for (size_t kw = 0; kw < kernel_size_.second; ++kw) {
                                int ih = static_cast<int>(oh * stride_.first + kh) - 
                                         static_cast<int>(padding_.first);
                                int iw = static_cast<int>(ow * stride_.second + kw) - 
                                         static_cast<int>(padding_.second);
                                
                                if (ih >= 0 && ih < static_cast<int>(in_h) &&
                                    iw >= 0 && iw < static_cast<int>(in_w)) {
                                    float val = x->data[b * channels * in_h * in_w + 
                                                       c * in_h * in_w + ih * in_w + iw];
                                    max_val = std::max(max_val, val);
                                }
                            }
                        }
                        
                        output[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = max_val;
                    }
                }
            }
        }
        
        return std::make_shared<Tensor>(output, {batch, channels, out_h, out_w}, x->requires_grad);
    }
};

/**
 * @brief 3D Max Pooling
 */
class MaxPool3d : public Module {
private:
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    
public:
    MaxPool3d(size_t kernel_size, size_t stride = 0, size_t padding = 0)
        : kernel_size_(kernel_size), stride_(stride == 0 ? kernel_size : stride),
          padding_(padding) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        return x;  // Placeholder
    }
};

/**
 * @brief 1D Average Pooling
 */
class AvgPool1d : public Module {
private:
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    bool count_include_pad_;
    
public:
    AvgPool1d(size_t kernel_size, size_t stride = 0, size_t padding = 0, 
              bool count_include_pad = true)
        : kernel_size_(kernel_size), stride_(stride == 0 ? kernel_size : stride),
          padding_(padding), count_include_pad_(count_include_pad) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t in_len = x->shape[2];
        
        size_t out_len = (in_len + 2 * padding_ - kernel_size_) / stride_ + 1;
        
        std::vector<float> output(batch * channels * out_len);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t o = 0; o < out_len; ++o) {
                    float sum = 0.0f;
                    size_t count = 0;
                    
                    for (size_t k = 0; k < kernel_size_; ++k) {
                        int idx = static_cast<int>(o * stride_ + k) - static_cast<int>(padding_);
                        if (idx >= 0 && idx < static_cast<int>(in_len)) {
                            sum += x->data[b * channels * in_len + c * in_len + idx];
                            count++;
                        } else if (count_include_pad_) {
                            count++;
                        }
                    }
                    
                    output[b * channels * out_len + c * out_len + o] = sum / (count_include_pad_ ? kernel_size_ : count);
                }
            }
        }
        
        return std::make_shared<Tensor>(output, {batch, channels, out_len}, x->requires_grad);
    }
};

/**
 * @brief 2D Average Pooling
 */
class AvgPool2d : public Module {
private:
    std::pair<size_t, size_t> kernel_size_;
    std::pair<size_t, size_t> stride_;
    std::pair<size_t, size_t> padding_;
    
public:
    AvgPool2d(size_t kernel_size, size_t stride = 0, size_t padding = 0)
        : kernel_size_({kernel_size, kernel_size}),
          stride_({stride == 0 ? kernel_size : stride, stride == 0 ? kernel_size : stride}),
          padding_({padding, padding}) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t in_h = x->shape[2];
        size_t in_w = x->shape[3];
        
        size_t out_h = (in_h + 2 * padding_.first - kernel_size_.first) / stride_.first + 1;
        size_t out_w = (in_w + 2 * padding_.second - kernel_size_.second) / stride_.second + 1;
        
        std::vector<float> output(batch * channels * out_h * out_w);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        float sum = 0.0f;
                        size_t count = 0;
                        
                        for (size_t kh = 0; kh < kernel_size_.first; ++kh) {
                            for (size_t kw = 0; kw < kernel_size_.second; ++kw) {
                                int ih = static_cast<int>(oh * stride_.first + kh) - 
                                         static_cast<int>(padding_.first);
                                int iw = static_cast<int>(ow * stride_.second + kw) - 
                                         static_cast<int>(padding_.second);
                                
                                if (ih >= 0 && ih < static_cast<int>(in_h) &&
                                    iw >= 0 && iw < static_cast<int>(in_w)) {
                                    sum += x->data[b * channels * in_h * in_w + 
                                                  c * in_h * in_w + ih * in_w + iw];
                                    count++;
                                }
                            }
                        }
                        
                        output[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = 
                            sum / (kernel_size_.first * kernel_size_.second);
                    }
                }
            }
        }
        
        return std::make_shared<Tensor>(output, {batch, channels, out_h, out_w}, x->requires_grad);
    }
};

/**
 * @brief 3D Average Pooling
 */
class AvgPool3d : public Module {
private:
    size_t kernel_size_;
    size_t stride_;
    
public:
    AvgPool3d(size_t kernel_size, size_t stride = 0)
        : kernel_size_(kernel_size), stride_(stride == 0 ? kernel_size : stride) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        return x;  // Placeholder
    }
};

/**
 * @brief Adaptive 1D Average Pooling
 */
class AdaptiveAvgPool1d : public Module {
private:
    size_t output_size_;
    
public:
    explicit AdaptiveAvgPool1d(size_t output_size) : output_size_(output_size) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t in_len = x->shape[2];
        
        std::vector<float> output(batch * channels * output_size_);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t o = 0; o < output_size_; ++o) {
                    size_t start = o * in_len / output_size_;
                    size_t end = (o + 1) * in_len / output_size_;
                    
                    float sum = 0.0f;
                    for (size_t i = start; i < end; ++i) {
                        sum += x->data[b * channels * in_len + c * in_len + i];
                    }
                    
                    output[b * channels * output_size_ + c * output_size_ + o] = sum / (end - start);
                }
            }
        }
        
        return std::make_shared<Tensor>(output, {batch, channels, output_size_}, x->requires_grad);
    }
};

/**
 * @brief Adaptive 2D Average Pooling
 */
class AdaptiveAvgPool2d : public Module {
private:
    std::pair<size_t, size_t> output_size_;
    
public:
    explicit AdaptiveAvgPool2d(size_t output_size) 
        : output_size_({output_size, output_size}) {}
    
    AdaptiveAvgPool2d(std::pair<size_t, size_t> output_size)
        : output_size_(output_size) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t in_h = x->shape[2];
        size_t in_w = x->shape[3];
        
        size_t out_h = output_size_.first;
        size_t out_w = output_size_.second;
        
        std::vector<float> output(batch * channels * out_h * out_w);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        size_t h_start = oh * in_h / out_h;
                        size_t h_end = (oh + 1) * in_h / out_h;
                        size_t w_start = ow * in_w / out_w;
                        size_t w_end = (ow + 1) * in_w / out_w;
                        
                        float sum = 0.0f;
                        for (size_t ih = h_start; ih < h_end; ++ih) {
                            for (size_t iw = w_start; iw < w_end; ++iw) {
                                sum += x->data[b * channels * in_h * in_w + 
                                              c * in_h * in_w + ih * in_w + iw];
                            }
                        }
                        
                        output[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = 
                            sum / ((h_end - h_start) * (w_end - w_start));
                    }
                }
            }
        }
        
        return std::make_shared<Tensor>(output, {batch, channels, out_h, out_w}, x->requires_grad);
    }
};

/**
 * @brief Adaptive 1D Max Pooling
 */
class AdaptiveMaxPool1d : public Module {
private:
    size_t output_size_;
    
public:
    explicit AdaptiveMaxPool1d(size_t output_size) : output_size_(output_size) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t in_len = x->shape[2];
        
        std::vector<float> output(batch * channels * output_size_);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t o = 0; o < output_size_; ++o) {
                    size_t start = o * in_len / output_size_;
                    size_t end = (o + 1) * in_len / output_size_;
                    
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (size_t i = start; i < end; ++i) {
                        max_val = std::max(max_val, x->data[b * channels * in_len + c * in_len + i]);
                    }
                    
                    output[b * channels * output_size_ + c * output_size_ + o] = max_val;
                }
            }
        }
        
        return std::make_shared<Tensor>(output, {batch, channels, output_size_}, x->requires_grad);
    }
};

/**
 * @brief Adaptive 2D Max Pooling
 */
class AdaptiveMaxPool2d : public Module {
private:
    std::pair<size_t, size_t> output_size_;
    
public:
    explicit AdaptiveMaxPool2d(size_t output_size) 
        : output_size_({output_size, output_size}) {}
    
    AdaptiveMaxPool2d(std::pair<size_t, size_t> output_size)
        : output_size_(output_size) {}
    
    TensorPtr forward(const TensorPtr& x) override {
        size_t batch = x->shape[0];
        size_t channels = x->shape[1];
        size_t in_h = x->shape[2];
        size_t in_w = x->shape[3];
        
        size_t out_h = output_size_.first;
        size_t out_w = output_size_.second;
        
        std::vector<float> output(batch * channels * out_h * out_w);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        size_t h_start = oh * in_h / out_h;
                        size_t h_end = (oh + 1) * in_h / out_h;
                        size_t w_start = ow * in_w / out_w;
                        size_t w_end = (ow + 1) * in_w / out_w;
                        
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (size_t ih = h_start; ih < h_end; ++ih) {
                            for (size_t iw = w_start; iw < w_end; ++iw) {
                                max_val = std::max(max_val, 
                                    x->data[b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw]);
                            }
                        }
                        
                        output[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = max_val;
                    }
                }
            }
        }
        
        return std::make_shared<Tensor>(output, {batch, channels, out_h, out_w}, x->requires_grad);
    }
};

/**
 * @brief Global Average Pooling (produces 1x1 output per channel)
 */
class GlobalAvgPool2d : public AdaptiveAvgPool2d {
public:
    GlobalAvgPool2d() : AdaptiveAvgPool2d({1, 1}) {}
};

/**
 * @brief Global Max Pooling
 */
class GlobalMaxPool2d : public AdaptiveMaxPool2d {
public:
    GlobalMaxPool2d() : AdaptiveMaxPool2d({1, 1}) {}
};

} // namespace nn
} // namespace neurova
