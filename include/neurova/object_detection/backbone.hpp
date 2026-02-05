// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file object_detection/backbone.hpp
 * @brief Feature extraction backbones for object detection
 * 
 * Neurova implementation of backbone networks like ResNet, VGG, etc.
 */

#pragma once

#include "../nn/layers.hpp"
#include "../nn/conv.hpp"
#include "../nn/pooling.hpp"
#include "../nn/normalization.hpp"
#include "../nn/activation.hpp"
#include <vector>
#include <memory>
#include <string>

namespace neurova {
namespace object_detection {

/**
 * @brief Basic convolution block with BatchNorm and ReLU
 */
class ConvBNReLU : public nn::Module {
public:
    ConvBNReLU(int in_channels, int out_channels, int kernel_size = 3,
               int stride = 1, int padding = 1, bool bias = false)
        : conv(in_channels, out_channels, kernel_size, stride, padding, 1, 1, bias),
          bn(out_channels),
          relu() {}
    
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& x) {
        auto out = conv.forward(x);
        out = bn.forward(out);
        return relu.forward(out);
    }
    
private:
    nn::Conv2d conv;
    nn::BatchNorm2d bn;
    nn::ReLU relu;
};

/**
 * @brief Residual block for ResNet
 */
class ResidualBlock : public nn::Module {
public:
    ResidualBlock(int in_channels, int out_channels, int stride = 1)
        : conv1(in_channels, out_channels, 3, stride, 1, 1, 1, false),
          bn1(out_channels),
          conv2(out_channels, out_channels, 3, 1, 1, 1, 1, false),
          bn2(out_channels),
          relu(),
          has_downsample(stride != 1 || in_channels != out_channels) {
        
        if (has_downsample) {
            downsample_conv = std::make_unique<nn::Conv2d>(in_channels, out_channels, 1, stride, 0, 1, 1, false);
            downsample_bn = std::make_unique<nn::BatchNorm2d>(out_channels);
        }
    }
    
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& x) {
        
        auto identity = x;
        
        auto out = conv1.forward(x);
        out = bn1.forward(out);
        out = relu.forward(out);
        
        out = conv2.forward(out);
        out = bn2.forward(out);
        
        if (has_downsample) {
            identity = downsample_conv->forward(identity);
            identity = downsample_bn->forward(identity);
        }
        
        // Add residual
        for (size_t c = 0; c < out.size(); ++c) {
            for (size_t h = 0; h < out[0].size(); ++h) {
                for (size_t w = 0; w < out[0][0].size(); ++w) {
                    out[c][h][w] += identity[c][h][w];
                }
            }
        }
        
        return relu.forward(out);
    }
    
private:
    nn::Conv2d conv1, conv2;
    nn::BatchNorm2d bn1, bn2;
    nn::ReLU relu;
    bool has_downsample;
    std::unique_ptr<nn::Conv2d> downsample_conv;
    std::unique_ptr<nn::BatchNorm2d> downsample_bn;
};

/**
 * @brief Bottleneck block for deeper ResNets
 */
class Bottleneck : public nn::Module {
public:
    static constexpr int expansion = 4;
    
    Bottleneck(int in_channels, int planes, int stride = 1)
        : conv1(in_channels, planes, 1, 1, 0, 1, 1, false),
          bn1(planes),
          conv2(planes, planes, 3, stride, 1, 1, 1, false),
          bn2(planes),
          conv3(planes, planes * expansion, 1, 1, 0, 1, 1, false),
          bn3(planes * expansion),
          relu(),
          has_downsample(stride != 1 || in_channels != planes * expansion) {
        
        if (has_downsample) {
            downsample_conv = std::make_unique<nn::Conv2d>(
                in_channels, planes * expansion, 1, stride, 0, 1, 1, false);
            downsample_bn = std::make_unique<nn::BatchNorm2d>(planes * expansion);
        }
    }
    
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& x) {
        
        auto identity = x;
        
        auto out = conv1.forward(x);
        out = bn1.forward(out);
        out = relu.forward(out);
        
        out = conv2.forward(out);
        out = bn2.forward(out);
        out = relu.forward(out);
        
        out = conv3.forward(out);
        out = bn3.forward(out);
        
        if (has_downsample) {
            identity = downsample_conv->forward(identity);
            identity = downsample_bn->forward(identity);
        }
        
        // Add residual
        for (size_t c = 0; c < out.size(); ++c) {
            for (size_t h = 0; h < out[0].size(); ++h) {
                for (size_t w = 0; w < out[0][0].size(); ++w) {
                    out[c][h][w] += identity[c][h][w];
                }
            }
        }
        
        return relu.forward(out);
    }
    
private:
    nn::Conv2d conv1, conv2, conv3;
    nn::BatchNorm2d bn1, bn2, bn3;
    nn::ReLU relu;
    bool has_downsample;
    std::unique_ptr<nn::Conv2d> downsample_conv;
    std::unique_ptr<nn::BatchNorm2d> downsample_bn;
};

/**
 * @brief Feature Pyramid Network (FPN)
 */
class FPN : public nn::Module {
public:
    FPN(const std::vector<int>& in_channels, int out_channels)
        : out_channels_(out_channels) {
        
        // Lateral connections (1x1 conv)
        for (int in_ch : in_channels) {
            lateral_convs.push_back(
                std::make_unique<nn::Conv2d>(in_ch, out_channels, 1, 1, 0, 1, 1, true));
        }
        
        // Output convolutions (3x3 conv)
        for (size_t i = 0; i < in_channels.size(); ++i) {
            output_convs.push_back(
                std::make_unique<nn::Conv2d>(out_channels, out_channels, 3, 1, 1, 1, 1, true));
        }
    }
    
    std::vector<std::vector<std::vector<std::vector<float>>>> forward(
        const std::vector<std::vector<std::vector<std::vector<float>>>>& features) {
        
        std::vector<std::vector<std::vector<std::vector<float>>>> outputs(features.size());
        
        // Process from top to bottom
        for (int i = static_cast<int>(features.size()) - 1; i >= 0; --i) {
            auto lateral = lateral_convs[i]->forward(features[i]);
            
            if (i < static_cast<int>(features.size()) - 1) {
                // Upsample and add
                auto& upper = outputs[i + 1];
                // Simple 2x upsampling (nearest neighbor)
                for (size_t c = 0; c < lateral.size(); ++c) {
                    int h = lateral[0].size();
                    int w = lateral[0][0].size();
                    for (int y = 0; y < h; ++y) {
                        for (int x = 0; x < w; ++x) {
                            int uy = y / 2;
                            int ux = x / 2;
                            if (uy < static_cast<int>(upper[c].size()) && 
                                ux < static_cast<int>(upper[c][0].size())) {
                                lateral[c][y][x] += upper[c][uy][ux];
                            }
                        }
                    }
                }
            }
            
            outputs[i] = output_convs[i]->forward(lateral);
        }
        
        return outputs;
    }
    
private:
    int out_channels_;
    std::vector<std::unique_ptr<nn::Conv2d>> lateral_convs;
    std::vector<std::unique_ptr<nn::Conv2d>> output_convs;
};

/**
 * @brief CSP (Cross Stage Partial) block for feature extraction
 */
class CSPBlock : public nn::Module {
public:
    CSPBlock(int in_channels, int out_channels, int num_blocks = 1)
        : conv1(in_channels, out_channels / 2, 1, 1, 0, 1, 1, false),
          bn1(out_channels / 2),
          conv2(in_channels, out_channels / 2, 1, 1, 0, 1, 1, false),
          bn2(out_channels / 2),
          conv3(out_channels, out_channels, 1, 1, 0, 1, 1, false),
          bn3(out_channels),
          relu() {
        
        int hidden_channels = out_channels / 2;
        for (int i = 0; i < num_blocks; ++i) {
            blocks.push_back(std::make_unique<ResidualBlock>(hidden_channels, hidden_channels));
        }
    }
    
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& x) {
        
        auto route1 = conv1.forward(x);
        route1 = bn1.forward(route1);
        route1 = relu.forward(route1);
        
        auto route2 = conv2.forward(x);
        route2 = bn2.forward(route2);
        route2 = relu.forward(route2);
        
        for (auto& block : blocks) {
            route2 = block->forward(route2);
        }
        
        // Concatenate
        std::vector<std::vector<std::vector<float>>> concat;
        concat.reserve(route1.size() + route2.size());
        for (auto& ch : route1) concat.push_back(std::move(ch));
        for (auto& ch : route2) concat.push_back(std::move(ch));
        
        auto out = conv3.forward(concat);
        out = bn3.forward(out);
        return relu.forward(out);
    }
    
private:
    nn::Conv2d conv1, conv2, conv3;
    nn::BatchNorm2d bn1, bn2, bn3;
    nn::ReLU relu;
    std::vector<std::unique_ptr<ResidualBlock>> blocks;
};

/**
 * @brief SPP (Spatial Pyramid Pooling) block
 */
class SPP : public nn::Module {
public:
    SPP(int in_channels, int out_channels, const std::vector<int>& kernel_sizes = {5, 9, 13})
        : conv1(in_channels, in_channels / 2, 1, 1, 0, 1, 1, false),
          bn1(in_channels / 2),
          conv2(in_channels / 2 * (kernel_sizes.size() + 1), out_channels, 1, 1, 0, 1, 1, false),
          bn2(out_channels),
          relu() {
        
        for (int k : kernel_sizes) {
            pools.push_back(std::make_unique<nn::MaxPool2d>(k, 1, k / 2));
        }
    }
    
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& x) {
        
        auto out = conv1.forward(x);
        out = bn1.forward(out);
        out = relu.forward(out);
        
        std::vector<std::vector<std::vector<float>>> concat = out;
        
        for (auto& pool : pools) {
            auto pooled = pool->forward(out);
            for (auto& ch : pooled) {
                concat.push_back(std::move(ch));
            }
        }
        
        auto result = conv2.forward(concat);
        result = bn2.forward(result);
        return relu.forward(result);
    }
    
private:
    nn::Conv2d conv1, conv2;
    nn::BatchNorm2d bn1, bn2;
    nn::ReLU relu;
    std::vector<std::unique_ptr<nn::MaxPool2d>> pools;
};

/**
 * @brief PANet (Path Aggregation Network) neck
 */
class PANet : public nn::Module {
public:
    PANet(const std::vector<int>& in_channels, int out_channels)
        : fpn_(in_channels, out_channels), out_channels_(out_channels) {
        
        // Bottom-up path
        for (size_t i = 0; i < in_channels.size() - 1; ++i) {
            downsample_convs.push_back(
                std::make_unique<nn::Conv2d>(out_channels, out_channels, 3, 2, 1, 1, 1, false));
            downsample_bns.push_back(
                std::make_unique<nn::BatchNorm2d>(out_channels));
            
            lateral_convs.push_back(
                std::make_unique<nn::Conv2d>(out_channels * 2, out_channels, 1, 1, 0, 1, 1, false));
            lateral_bns.push_back(
                std::make_unique<nn::BatchNorm2d>(out_channels));
        }
    }
    
    std::vector<std::vector<std::vector<std::vector<float>>>> forward(
        const std::vector<std::vector<std::vector<std::vector<float>>>>& features) {
        
        // Top-down (FPN)
        auto fpn_features = fpn_.forward(features);
        
        // Bottom-up
        std::vector<std::vector<std::vector<std::vector<float>>>> outputs;
        outputs.push_back(fpn_features[0]);
        
        nn::ReLU relu;
        for (size_t i = 0; i < downsample_convs.size(); ++i) {
            auto down = downsample_convs[i]->forward(outputs.back());
            down = downsample_bns[i]->forward(down);
            down = relu.forward(down);
            
            // Concatenate with FPN feature
            auto& fpn_feat = fpn_features[i + 1];
            std::vector<std::vector<std::vector<float>>> concat;
            for (auto& ch : down) concat.push_back(std::move(ch));
            for (auto& ch : fpn_feat) concat.push_back(ch);
            
            auto lateral = lateral_convs[i]->forward(concat);
            lateral = lateral_bns[i]->forward(lateral);
            lateral = relu.forward(lateral);
            
            outputs.push_back(lateral);
        }
        
        return outputs;
    }
    
private:
    FPN fpn_;
    int out_channels_;
    std::vector<std::unique_ptr<nn::Conv2d>> downsample_convs;
    std::vector<std::unique_ptr<nn::BatchNorm2d>> downsample_bns;
    std::vector<std::unique_ptr<nn::Conv2d>> lateral_convs;
    std::vector<std::unique_ptr<nn::BatchNorm2d>> lateral_bns;
};

/**
 * @brief Depthwise separable convolution (MobileNet-style)
 */
class DepthwiseSeparableConv : public nn::Module {
public:
    DepthwiseSeparableConv(int in_channels, int out_channels, int kernel_size = 3,
                           int stride = 1, int padding = 1)
        : depthwise(in_channels, in_channels, kernel_size, stride, padding, 1, in_channels, false),
          bn1(in_channels),
          pointwise(in_channels, out_channels, 1, 1, 0, 1, 1, false),
          bn2(out_channels),
          relu() {}
    
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& x) {
        
        auto out = depthwise.forward(x);
        out = bn1.forward(out);
        out = relu.forward(out);
        
        out = pointwise.forward(out);
        out = bn2.forward(out);
        return relu.forward(out);
    }
    
private:
    nn::Conv2d depthwise, pointwise;
    nn::BatchNorm2d bn1, bn2;
    nn::ReLU relu;
};

/**
 * @brief Inverted residual block (MobileNetV2-style)
 */
class InvertedResidual : public nn::Module {
public:
    InvertedResidual(int in_channels, int out_channels, int stride = 1, int expand_ratio = 6)
        : use_residual(stride == 1 && in_channels == out_channels) {
        
        int hidden = in_channels * expand_ratio;
        
        if (expand_ratio != 1) {
            expand_conv = std::make_unique<nn::Conv2d>(in_channels, hidden, 1, 1, 0, 1, 1, false);
            expand_bn = std::make_unique<nn::BatchNorm2d>(hidden);
        }
        
        depthwise = std::make_unique<nn::Conv2d>(hidden, hidden, 3, stride, 1, 1, hidden, false);
        dw_bn = std::make_unique<nn::BatchNorm2d>(hidden);
        
        project = std::make_unique<nn::Conv2d>(hidden, out_channels, 1, 1, 0, 1, 1, false);
        project_bn = std::make_unique<nn::BatchNorm2d>(out_channels);
    }
    
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<std::vector<std::vector<float>>>& x) {
        
        nn::ReLU6 relu6;
        auto out = x;
        
        if (expand_conv) {
            out = expand_conv->forward(out);
            out = expand_bn->forward(out);
            out = relu6.forward(out);
        }
        
        out = depthwise->forward(out);
        out = dw_bn->forward(out);
        out = relu6.forward(out);
        
        out = project->forward(out);
        out = project_bn->forward(out);
        
        if (use_residual) {
            for (size_t c = 0; c < out.size(); ++c) {
                for (size_t h = 0; h < out[0].size(); ++h) {
                    for (size_t w = 0; w < out[0][0].size(); ++w) {
                        out[c][h][w] += x[c][h][w];
                    }
                }
            }
        }
        
        return out;
    }
    
private:
    bool use_residual;
    std::unique_ptr<nn::Conv2d> expand_conv;
    std::unique_ptr<nn::BatchNorm2d> expand_bn;
    std::unique_ptr<nn::Conv2d> depthwise;
    std::unique_ptr<nn::BatchNorm2d> dw_bn;
    std::unique_ptr<nn::Conv2d> project;
    std::unique_ptr<nn::BatchNorm2d> project_bn;
};

} // namespace object_detection
} // namespace neurova
