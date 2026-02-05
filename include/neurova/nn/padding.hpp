// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file padding.hpp
 * @brief Padding layers for neural networks
 * 
 * Neurova implementation of various padding modules.
 */

#pragma once

#include "tensor.hpp"
#include "layers.hpp"
#include <algorithm>

namespace neurova {
namespace nn {

/**
 * @brief Constant padding (1D)
 */
class ConstantPad1d : public Module {
private:
    int padding_left_;
    int padding_right_;
    float value_;
    
public:
    ConstantPad1d(int padding, float value = 0.0f)
        : padding_left_(padding), padding_right_(padding), value_(value) {}
    
    ConstantPad1d(std::pair<int, int> padding, float value = 0.0f)
        : padding_left_(padding.first), padding_right_(padding.second), value_(value) {}
    
    Tensor forward(const Tensor& input) override {
        auto shape = input.shape();
        int in_length = shape.back();
        int out_length = in_length + padding_left_ + padding_right_;
        
        // Calculate batch dimensions
        int batch_size = 1;
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            batch_size *= shape[i];
        }
        
        std::vector<float> result(batch_size * out_length, value_);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < in_length; ++i) {
                result[b * out_length + padding_left_ + i] = input.data()[b * in_length + i];
            }
        }
        
        std::vector<int> new_shape = shape;
        new_shape.back() = out_length;
        
        return Tensor(result, new_shape);
    }
};

/**
 * @brief Constant padding (2D)
 */
class ConstantPad2d : public Module {
private:
    int pad_left_, pad_right_, pad_top_, pad_bottom_;
    float value_;
    
public:
    ConstantPad2d(int padding, float value = 0.0f)
        : pad_left_(padding), pad_right_(padding), 
          pad_top_(padding), pad_bottom_(padding), value_(value) {}
    
    ConstantPad2d(std::tuple<int, int, int, int> padding, float value = 0.0f)
        : pad_left_(std::get<0>(padding)), pad_right_(std::get<1>(padding)),
          pad_top_(std::get<2>(padding)), pad_bottom_(std::get<3>(padding)), value_(value) {}
    
    Tensor forward(const Tensor& input) override {
        auto shape = input.shape();
        int H = shape[shape.size() - 2];
        int W = shape[shape.size() - 1];
        int new_H = H + pad_top_ + pad_bottom_;
        int new_W = W + pad_left_ + pad_right_;
        
        int batch_size = 1;
        for (size_t i = 0; i < shape.size() - 2; ++i) {
            batch_size *= shape[i];
        }
        
        std::vector<float> result(batch_size * new_H * new_W, value_);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int src_idx = b * H * W + h * W + w;
                    int dst_idx = b * new_H * new_W + (h + pad_top_) * new_W + (w + pad_left_);
                    result[dst_idx] = input.data()[src_idx];
                }
            }
        }
        
        std::vector<int> new_shape = shape;
        new_shape[shape.size() - 2] = new_H;
        new_shape[shape.size() - 1] = new_W;
        
        return Tensor(result, new_shape);
    }
};

/**
 * @brief Constant padding (3D)
 */
class ConstantPad3d : public Module {
private:
    std::array<int, 6> padding_;  // left, right, top, bottom, front, back
    float value_;
    
public:
    ConstantPad3d(int padding, float value = 0.0f) : value_(value) {
        padding_.fill(padding);
    }
    
    ConstantPad3d(std::array<int, 6> padding, float value = 0.0f)
        : padding_(padding), value_(value) {}
    
    Tensor forward(const Tensor& input) override {
        auto shape = input.shape();
        int D = shape[shape.size() - 3];
        int H = shape[shape.size() - 2];
        int W = shape[shape.size() - 1];
        
        int new_D = D + padding_[4] + padding_[5];
        int new_H = H + padding_[2] + padding_[3];
        int new_W = W + padding_[0] + padding_[1];
        
        int batch_size = 1;
        for (size_t i = 0; i < shape.size() - 3; ++i) {
            batch_size *= shape[i];
        }
        
        std::vector<float> result(batch_size * new_D * new_H * new_W, value_);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int d = 0; d < D; ++d) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int src_idx = b * D * H * W + d * H * W + h * W + w;
                        int dst_idx = b * new_D * new_H * new_W + 
                                      (d + padding_[4]) * new_H * new_W +
                                      (h + padding_[2]) * new_W + 
                                      (w + padding_[0]);
                        result[dst_idx] = input.data()[src_idx];
                    }
                }
            }
        }
        
        std::vector<int> new_shape = shape;
        new_shape[shape.size() - 3] = new_D;
        new_shape[shape.size() - 2] = new_H;
        new_shape[shape.size() - 1] = new_W;
        
        return Tensor(result, new_shape);
    }
};

/**
 * @brief Reflection padding (1D)
 */
class ReflectionPad1d : public Module {
private:
    int padding_left_;
    int padding_right_;
    
public:
    ReflectionPad1d(int padding)
        : padding_left_(padding), padding_right_(padding) {}
    
    ReflectionPad1d(std::pair<int, int> padding)
        : padding_left_(padding.first), padding_right_(padding.second) {}
    
    Tensor forward(const Tensor& input) override {
        auto shape = input.shape();
        int in_length = shape.back();
        int out_length = in_length + padding_left_ + padding_right_;
        
        int batch_size = 1;
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            batch_size *= shape[i];
        }
        
        std::vector<float> result(batch_size * out_length);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < out_length; ++i) {
                int src_idx;
                if (i < padding_left_) {
                    // Left padding: reflect
                    src_idx = padding_left_ - i;
                } else if (i >= in_length + padding_left_) {
                    // Right padding: reflect
                    src_idx = 2 * in_length + padding_left_ - i - 2;
                } else {
                    src_idx = i - padding_left_;
                }
                src_idx = std::max(0, std::min(in_length - 1, src_idx));
                result[b * out_length + i] = input.data()[b * in_length + src_idx];
            }
        }
        
        std::vector<int> new_shape = shape;
        new_shape.back() = out_length;
        
        return Tensor(result, new_shape);
    }
};

/**
 * @brief Reflection padding (2D)
 */
class ReflectionPad2d : public Module {
private:
    int pad_left_, pad_right_, pad_top_, pad_bottom_;
    
public:
    ReflectionPad2d(int padding)
        : pad_left_(padding), pad_right_(padding),
          pad_top_(padding), pad_bottom_(padding) {}
    
    ReflectionPad2d(std::tuple<int, int, int, int> padding)
        : pad_left_(std::get<0>(padding)), pad_right_(std::get<1>(padding)),
          pad_top_(std::get<2>(padding)), pad_bottom_(std::get<3>(padding)) {}
    
    Tensor forward(const Tensor& input) override {
        auto shape = input.shape();
        int H = shape[shape.size() - 2];
        int W = shape[shape.size() - 1];
        int new_H = H + pad_top_ + pad_bottom_;
        int new_W = W + pad_left_ + pad_right_;
        
        int batch_size = 1;
        for (size_t i = 0; i < shape.size() - 2; ++i) {
            batch_size *= shape[i];
        }
        
        std::vector<float> result(batch_size * new_H * new_W);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < new_H; ++h) {
                for (int w = 0; w < new_W; ++w) {
                    int src_h, src_w;
                    
                    // Reflect height
                    if (h < pad_top_) {
                        src_h = pad_top_ - h;
                    } else if (h >= H + pad_top_) {
                        src_h = 2 * H + pad_top_ - h - 2;
                    } else {
                        src_h = h - pad_top_;
                    }
                    
                    // Reflect width
                    if (w < pad_left_) {
                        src_w = pad_left_ - w;
                    } else if (w >= W + pad_left_) {
                        src_w = 2 * W + pad_left_ - w - 2;
                    } else {
                        src_w = w - pad_left_;
                    }
                    
                    src_h = std::max(0, std::min(H - 1, src_h));
                    src_w = std::max(0, std::min(W - 1, src_w));
                    
                    result[b * new_H * new_W + h * new_W + w] = 
                        input.data()[b * H * W + src_h * W + src_w];
                }
            }
        }
        
        std::vector<int> new_shape = shape;
        new_shape[shape.size() - 2] = new_H;
        new_shape[shape.size() - 1] = new_W;
        
        return Tensor(result, new_shape);
    }
};

/**
 * @brief Replication padding (1D)
 */
class ReplicationPad1d : public Module {
private:
    int padding_left_;
    int padding_right_;
    
public:
    ReplicationPad1d(int padding)
        : padding_left_(padding), padding_right_(padding) {}
    
    ReplicationPad1d(std::pair<int, int> padding)
        : padding_left_(padding.first), padding_right_(padding.second) {}
    
    Tensor forward(const Tensor& input) override {
        auto shape = input.shape();
        int in_length = shape.back();
        int out_length = in_length + padding_left_ + padding_right_;
        
        int batch_size = 1;
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            batch_size *= shape[i];
        }
        
        std::vector<float> result(batch_size * out_length);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < out_length; ++i) {
                int src_idx = std::max(0, std::min(in_length - 1, i - padding_left_));
                result[b * out_length + i] = input.data()[b * in_length + src_idx];
            }
        }
        
        std::vector<int> new_shape = shape;
        new_shape.back() = out_length;
        
        return Tensor(result, new_shape);
    }
};

/**
 * @brief Replication padding (2D)
 */
class ReplicationPad2d : public Module {
private:
    int pad_left_, pad_right_, pad_top_, pad_bottom_;
    
public:
    ReplicationPad2d(int padding)
        : pad_left_(padding), pad_right_(padding),
          pad_top_(padding), pad_bottom_(padding) {}
    
    ReplicationPad2d(std::tuple<int, int, int, int> padding)
        : pad_left_(std::get<0>(padding)), pad_right_(std::get<1>(padding)),
          pad_top_(std::get<2>(padding)), pad_bottom_(std::get<3>(padding)) {}
    
    Tensor forward(const Tensor& input) override {
        auto shape = input.shape();
        int H = shape[shape.size() - 2];
        int W = shape[shape.size() - 1];
        int new_H = H + pad_top_ + pad_bottom_;
        int new_W = W + pad_left_ + pad_right_;
        
        int batch_size = 1;
        for (size_t i = 0; i < shape.size() - 2; ++i) {
            batch_size *= shape[i];
        }
        
        std::vector<float> result(batch_size * new_H * new_W);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < new_H; ++h) {
                for (int w = 0; w < new_W; ++w) {
                    int src_h = std::max(0, std::min(H - 1, h - pad_top_));
                    int src_w = std::max(0, std::min(W - 1, w - pad_left_));
                    
                    result[b * new_H * new_W + h * new_W + w] = 
                        input.data()[b * H * W + src_h * W + src_w];
                }
            }
        }
        
        std::vector<int> new_shape = shape;
        new_shape[shape.size() - 2] = new_H;
        new_shape[shape.size() - 1] = new_W;
        
        return Tensor(result, new_shape);
    }
};

/**
 * @brief Replication padding (3D)
 */
class ReplicationPad3d : public Module {
private:
    std::array<int, 6> padding_;
    
public:
    ReplicationPad3d(int padding) {
        padding_.fill(padding);
    }
    
    ReplicationPad3d(std::array<int, 6> padding) : padding_(padding) {}
    
    Tensor forward(const Tensor& input) override {
        auto shape = input.shape();
        int D = shape[shape.size() - 3];
        int H = shape[shape.size() - 2];
        int W = shape[shape.size() - 1];
        
        int new_D = D + padding_[4] + padding_[5];
        int new_H = H + padding_[2] + padding_[3];
        int new_W = W + padding_[0] + padding_[1];
        
        int batch_size = 1;
        for (size_t i = 0; i < shape.size() - 3; ++i) {
            batch_size *= shape[i];
        }
        
        std::vector<float> result(batch_size * new_D * new_H * new_W);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int d = 0; d < new_D; ++d) {
                for (int h = 0; h < new_H; ++h) {
                    for (int w = 0; w < new_W; ++w) {
                        int src_d = std::max(0, std::min(D - 1, d - padding_[4]));
                        int src_h = std::max(0, std::min(H - 1, h - padding_[2]));
                        int src_w = std::max(0, std::min(W - 1, w - padding_[0]));
                        
                        result[b * new_D * new_H * new_W + d * new_H * new_W + h * new_W + w] = 
                            input.data()[b * D * H * W + src_d * H * W + src_h * W + src_w];
                    }
                }
            }
        }
        
        std::vector<int> new_shape = shape;
        new_shape[shape.size() - 3] = new_D;
        new_shape[shape.size() - 2] = new_H;
        new_shape[shape.size() - 1] = new_W;
        
        return Tensor(result, new_shape);
    }
};

/**
 * @brief Zero padding (1D) - alias for ConstantPad1d with value=0
 */
class ZeroPad1d : public ConstantPad1d {
public:
    ZeroPad1d(int padding) : ConstantPad1d(padding, 0.0f) {}
    ZeroPad1d(std::pair<int, int> padding) : ConstantPad1d(padding, 0.0f) {}
};

/**
 * @brief Zero padding (2D) - alias for ConstantPad2d with value=0
 */
class ZeroPad2d : public ConstantPad2d {
public:
    ZeroPad2d(int padding) : ConstantPad2d(padding, 0.0f) {}
    ZeroPad2d(std::tuple<int, int, int, int> padding) : ConstantPad2d(padding, 0.0f) {}
};

/**
 * @brief Zero padding (3D) - alias for ConstantPad3d with value=0
 */
class ZeroPad3d : public ConstantPad3d {
public:
    ZeroPad3d(int padding) : ConstantPad3d(padding, 0.0f) {}
    ZeroPad3d(std::array<int, 6> padding) : ConstantPad3d(padding, 0.0f) {}
};

/**
 * @brief Circular padding (1D)
 */
class CircularPad1d : public Module {
private:
    int padding_left_;
    int padding_right_;
    
public:
    CircularPad1d(int padding)
        : padding_left_(padding), padding_right_(padding) {}
    
    CircularPad1d(std::pair<int, int> padding)
        : padding_left_(padding.first), padding_right_(padding.second) {}
    
    Tensor forward(const Tensor& input) override {
        auto shape = input.shape();
        int in_length = shape.back();
        int out_length = in_length + padding_left_ + padding_right_;
        
        int batch_size = 1;
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            batch_size *= shape[i];
        }
        
        std::vector<float> result(batch_size * out_length);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < out_length; ++i) {
                int src_idx = ((i - padding_left_) % in_length + in_length) % in_length;
                result[b * out_length + i] = input.data()[b * in_length + src_idx];
            }
        }
        
        std::vector<int> new_shape = shape;
        new_shape.back() = out_length;
        
        return Tensor(result, new_shape);
    }
};

/**
 * @brief Circular padding (2D)
 */
class CircularPad2d : public Module {
private:
    int pad_left_, pad_right_, pad_top_, pad_bottom_;
    
public:
    CircularPad2d(int padding)
        : pad_left_(padding), pad_right_(padding),
          pad_top_(padding), pad_bottom_(padding) {}
    
    CircularPad2d(std::tuple<int, int, int, int> padding)
        : pad_left_(std::get<0>(padding)), pad_right_(std::get<1>(padding)),
          pad_top_(std::get<2>(padding)), pad_bottom_(std::get<3>(padding)) {}
    
    Tensor forward(const Tensor& input) override {
        auto shape = input.shape();
        int H = shape[shape.size() - 2];
        int W = shape[shape.size() - 1];
        int new_H = H + pad_top_ + pad_bottom_;
        int new_W = W + pad_left_ + pad_right_;
        
        int batch_size = 1;
        for (size_t i = 0; i < shape.size() - 2; ++i) {
            batch_size *= shape[i];
        }
        
        std::vector<float> result(batch_size * new_H * new_W);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < new_H; ++h) {
                for (int w = 0; w < new_W; ++w) {
                    int src_h = ((h - pad_top_) % H + H) % H;
                    int src_w = ((w - pad_left_) % W + W) % W;
                    
                    result[b * new_H * new_W + h * new_W + w] = 
                        input.data()[b * H * W + src_h * W + src_w];
                }
            }
        }
        
        std::vector<int> new_shape = shape;
        new_shape[shape.size() - 2] = new_H;
        new_shape[shape.size() - 1] = new_W;
        
        return Tensor(result, new_shape);
    }
};

} // namespace nn
} // namespace neurova
