// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file object_detection/detector.hpp
 * @brief Object detection models
 * 
 * Neurova implementation of object detection architectures.
 */

#pragma once

#include "anchor.hpp"
#include "nms.hpp"
#include "backbone.hpp"
#include "../nn.hpp"
#include <vector>
#include <memory>
#include <string>

namespace neurova {
namespace object_detection {

/**
 * @brief Base class for object detectors
 */
class Detector {
public:
    virtual ~Detector() = default;
    
    virtual std::vector<Detection> detect(
        const std::vector<std::vector<std::vector<float>>>& image,
        float confidence_threshold = 0.5f,
        float nms_threshold = 0.4f) = 0;
    
    void set_num_classes(int num_classes) { num_classes_ = num_classes; }
    int num_classes() const { return num_classes_; }
    
protected:
    int num_classes_ = 80;  // COCO classes by default
};

/**
 * @brief SSD (Single Shot Detector) head
 */
class SSDHead : public nn::Module {
public:
    SSDHead(const std::vector<int>& in_channels, int num_classes, 
            const std::vector<int>& num_anchors)
        : num_classes_(num_classes) {
        
        for (size_t i = 0; i < in_channels.size(); ++i) {
            // Classification head
            cls_heads.push_back(std::make_unique<nn::Conv2d>(
                in_channels[i], num_anchors[i] * num_classes, 3, 1, 1, 1, 1, true));
            
            // Regression head
            reg_heads.push_back(std::make_unique<nn::Conv2d>(
                in_channels[i], num_anchors[i] * 4, 3, 1, 1, 1, 1, true));
        }
    }
    
    struct Output {
        std::vector<std::vector<float>> class_scores;  // [N_anchors, num_classes]
        std::vector<std::vector<float>> box_deltas;    // [N_anchors, 4]
    };
    
    Output forward(const std::vector<std::vector<std::vector<std::vector<float>>>>& features) {
        Output output;
        
        for (size_t level = 0; level < features.size(); ++level) {
            auto cls_out = cls_heads[level]->forward(features[level]);
            auto reg_out = reg_heads[level]->forward(features[level]);
            
            // Flatten spatial dimensions
            int h = cls_out[0].size();
            int w = cls_out[0][0].size();
            int num_anchors = cls_out.size() / num_classes_;
            
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    for (int a = 0; a < num_anchors; ++a) {
                        // Class scores
                        std::vector<float> scores(num_classes_);
                        for (int c = 0; c < num_classes_; ++c) {
                            scores[c] = cls_out[a * num_classes_ + c][y][x];
                        }
                        output.class_scores.push_back(scores);
                        
                        // Box deltas
                        std::vector<float> delta(4);
                        for (int d = 0; d < 4; ++d) {
                            delta[d] = reg_out[a * 4 + d][y][x];
                        }
                        output.box_deltas.push_back(delta);
                    }
                }
            }
        }
        
        return output;
    }
    
private:
    int num_classes_;
    std::vector<std::unique_ptr<nn::Conv2d>> cls_heads;
    std::vector<std::unique_ptr<nn::Conv2d>> reg_heads;
};

/**
 * @brief RetinaNet-style detection head with focal loss
 */
class RetinaNetHead : public nn::Module {
public:
    RetinaNetHead(int in_channels, int num_classes, int num_anchors = 9, int num_convs = 4)
        : num_classes_(num_classes), num_anchors_(num_anchors) {
        
        // Shared tower
        for (int i = 0; i < num_convs; ++i) {
            cls_tower.push_back(std::make_unique<nn::Conv2d>(
                in_channels, in_channels, 3, 1, 1, 1, 1, true));
            cls_tower_bn.push_back(std::make_unique<nn::BatchNorm2d>(in_channels));
            
            box_tower.push_back(std::make_unique<nn::Conv2d>(
                in_channels, in_channels, 3, 1, 1, 1, 1, true));
            box_tower_bn.push_back(std::make_unique<nn::BatchNorm2d>(in_channels));
        }
        
        // Output layers
        cls_pred = std::make_unique<nn::Conv2d>(
            in_channels, num_anchors * num_classes, 3, 1, 1, 1, 1, true);
        box_pred = std::make_unique<nn::Conv2d>(
            in_channels, num_anchors * 4, 3, 1, 1, 1, 1, true);
    }
    
    struct Output {
        std::vector<std::vector<std::vector<float>>> cls_logits;  // Per level
        std::vector<std::vector<std::vector<float>>> box_deltas;  // Per level
    };
    
    Output forward(const std::vector<std::vector<std::vector<std::vector<float>>>>& features) {
        Output output;
        nn::ReLU relu;
        
        for (const auto& feat : features) {
            // Classification tower
            auto cls_feat = feat;
            for (size_t i = 0; i < cls_tower.size(); ++i) {
                cls_feat = cls_tower[i]->forward(cls_feat);
                cls_feat = cls_tower_bn[i]->forward(cls_feat);
                cls_feat = relu.forward(cls_feat);
            }
            auto cls_out = cls_pred->forward(cls_feat);
            
            // Box regression tower
            auto box_feat = feat;
            for (size_t i = 0; i < box_tower.size(); ++i) {
                box_feat = box_tower[i]->forward(box_feat);
                box_feat = box_tower_bn[i]->forward(box_feat);
                box_feat = relu.forward(box_feat);
            }
            auto box_out = box_pred->forward(box_feat);
            
            // Reshape to [H*W*A, num_classes] and [H*W*A, 4]
            // (Simplified: just store as-is)
            output.cls_logits.push_back(std::move(cls_out));
            output.box_deltas.push_back(std::move(box_out));
        }
        
        return output;
    }
    
private:
    int num_classes_;
    int num_anchors_;
    std::vector<std::unique_ptr<nn::Conv2d>> cls_tower;
    std::vector<std::unique_ptr<nn::BatchNorm2d>> cls_tower_bn;
    std::vector<std::unique_ptr<nn::Conv2d>> box_tower;
    std::vector<std::unique_ptr<nn::BatchNorm2d>> box_tower_bn;
    std::unique_ptr<nn::Conv2d> cls_pred;
    std::unique_ptr<nn::Conv2d> box_pred;
};

/**
 * @brief Grid-based detection head
 */
class GridHead : public nn::Module {
public:
    GridHead(int in_channels, int num_classes, int num_anchors = 3)
        : num_classes_(num_classes), num_anchors_(num_anchors) {
        
        // Output: (num_anchors * (5 + num_classes)) channels
        // 5 = (x, y, w, h, objectness)
        int out_channels = num_anchors * (5 + num_classes);
        conv = std::make_unique<nn::Conv2d>(in_channels, out_channels, 1, 1, 0, 1, 1, true);
    }
    
    struct Output {
        std::vector<std::vector<float>> boxes;       // [N, 4] (cx, cy, w, h)
        std::vector<float> objectness;               // [N]
        std::vector<std::vector<float>> class_probs; // [N, num_classes]
    };
    
    Output forward(const std::vector<std::vector<std::vector<float>>>& x, int stride,
                   const std::vector<std::pair<int, int>>& anchors) {
        auto out = conv->forward(x);
        
        int h = out[0].size();
        int w = out[0][0].size();
        int pred_size = 5 + num_classes_;
        
        Output output;
        
        for (int y = 0; y < h; ++y) {
            for (int x_pos = 0; x_pos < w; ++x_pos) {
                for (int a = 0; a < num_anchors_; ++a) {
                    int offset = a * pred_size;
                    
                    // Box prediction
                    float tx = sigmoid(out[offset + 0][y][x_pos]);
                    float ty = sigmoid(out[offset + 1][y][x_pos]);
                    float tw = out[offset + 2][y][x_pos];
                    float th = out[offset + 3][y][x_pos];
                    
                    float cx = (x_pos + tx) * stride;
                    float cy = (y + ty) * stride;
                    float bw = anchors[a].first * std::exp(tw);
                    float bh = anchors[a].second * std::exp(th);
                    
                    output.boxes.push_back({cx, cy, bw, bh});
                    
                    // Objectness
                    float obj = sigmoid(out[offset + 4][y][x_pos]);
                    output.objectness.push_back(obj);
                    
                    // Class probabilities
                    std::vector<float> cls_probs(num_classes_);
                    float sum_exp = 0;
                    for (int c = 0; c < num_classes_; ++c) {
                        cls_probs[c] = std::exp(out[offset + 5 + c][y][x_pos]);
                        sum_exp += cls_probs[c];
                    }
                    for (int c = 0; c < num_classes_; ++c) {
                        cls_probs[c] /= sum_exp;
                    }
                    output.class_probs.push_back(cls_probs);
                }
            }
        }
        
        return output;
    }
    
private:
    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    int num_classes_;
    int num_anchors_;
    std::unique_ptr<nn::Conv2d> conv;
};

/**
 * @brief Decoupled detection head
 * 
 * Separates classification and regression branches
 */
class DecoupledHead : public nn::Module {
public:
    DecoupledHead(int in_channels, int num_classes, int hidden_channels = 256)
        : num_classes_(num_classes) {
        
        // Stem
        stem = std::make_unique<ConvBNReLU>(in_channels, hidden_channels, 1, 1, 0);
        
        // Classification branch
        cls_conv1 = std::make_unique<ConvBNReLU>(hidden_channels, hidden_channels, 3, 1, 1);
        cls_conv2 = std::make_unique<ConvBNReLU>(hidden_channels, hidden_channels, 3, 1, 1);
        cls_pred = std::make_unique<nn::Conv2d>(hidden_channels, num_classes, 1, 1, 0, 1, 1, true);
        
        // Regression branch  
        reg_conv1 = std::make_unique<ConvBNReLU>(hidden_channels, hidden_channels, 3, 1, 1);
        reg_conv2 = std::make_unique<ConvBNReLU>(hidden_channels, hidden_channels, 3, 1, 1);
        reg_pred = std::make_unique<nn::Conv2d>(hidden_channels, 4, 1, 1, 0, 1, 1, true);
        
        // Objectness
        obj_pred = std::make_unique<nn::Conv2d>(hidden_channels, 1, 1, 1, 0, 1, 1, true);
    }
    
    struct Output {
        std::vector<std::vector<std::vector<float>>> cls_out;
        std::vector<std::vector<std::vector<float>>> reg_out;
        std::vector<std::vector<std::vector<float>>> obj_out;
    };
    
    Output forward(const std::vector<std::vector<std::vector<float>>>& x) {
        auto feat = stem->forward(x);
        
        // Classification branch
        auto cls_feat = cls_conv1->forward(feat);
        cls_feat = cls_conv2->forward(cls_feat);
        auto cls_out = cls_pred->forward(cls_feat);
        
        // Regression branch
        auto reg_feat = reg_conv1->forward(feat);
        reg_feat = reg_conv2->forward(reg_feat);
        auto reg_out = reg_pred->forward(reg_feat);
        auto obj_out = obj_pred->forward(reg_feat);
        
        return {cls_out, reg_out, obj_out};
    }
    
private:
    int num_classes_;
    std::unique_ptr<ConvBNReLU> stem;
    std::unique_ptr<ConvBNReLU> cls_conv1, cls_conv2;
    std::unique_ptr<nn::Conv2d> cls_pred;
    std::unique_ptr<ConvBNReLU> reg_conv1, reg_conv2;
    std::unique_ptr<nn::Conv2d> reg_pred;
    std::unique_ptr<nn::Conv2d> obj_pred;
};

/**
 * @brief FCOS (Fully Convolutional One-Stage) head
 * 
 * Anchor-free detection
 */
class FCOSHead : public nn::Module {
public:
    FCOSHead(int in_channels, int num_classes, int num_convs = 4)
        : num_classes_(num_classes) {
        
        // Shared towers
        for (int i = 0; i < num_convs; ++i) {
            cls_tower.push_back(std::make_unique<ConvBNReLU>(in_channels, in_channels, 3, 1, 1));
            box_tower.push_back(std::make_unique<ConvBNReLU>(in_channels, in_channels, 3, 1, 1));
        }
        
        // Output layers
        cls_pred = std::make_unique<nn::Conv2d>(in_channels, num_classes, 3, 1, 1, 1, 1, true);
        box_pred = std::make_unique<nn::Conv2d>(in_channels, 4, 3, 1, 1, 1, 1, true);  // l, t, r, b
        centerness = std::make_unique<nn::Conv2d>(in_channels, 1, 3, 1, 1, 1, 1, true);
    }
    
    struct Output {
        std::vector<std::vector<std::vector<float>>> cls_out;
        std::vector<std::vector<std::vector<float>>> box_out;  // (l, t, r, b)
        std::vector<std::vector<std::vector<float>>> centerness_out;
    };
    
    Output forward(const std::vector<std::vector<std::vector<float>>>& x) {
        auto cls_feat = x;
        for (auto& conv : cls_tower) {
            cls_feat = conv->forward(cls_feat);
        }
        
        auto box_feat = x;
        for (auto& conv : box_tower) {
            box_feat = conv->forward(box_feat);
        }
        
        auto cls_out = cls_pred->forward(cls_feat);
        auto box_out = box_pred->forward(box_feat);
        auto cent_out = centerness->forward(box_feat);
        
        // Apply exp to box predictions (they predict positive values)
        for (auto& ch : box_out) {
            for (auto& row : ch) {
                for (auto& val : row) {
                    val = std::exp(val);
                }
            }
        }
        
        return {cls_out, box_out, cent_out};
    }
    
private:
    int num_classes_;
    std::vector<std::unique_ptr<ConvBNReLU>> cls_tower;
    std::vector<std::unique_ptr<ConvBNReLU>> box_tower;
    std::unique_ptr<nn::Conv2d> cls_pred;
    std::unique_ptr<nn::Conv2d> box_pred;
    std::unique_ptr<nn::Conv2d> centerness;
};

/**
 * @brief Post-processing utilities
 */
namespace postprocess {

/**
 * @brief Apply sigmoid activation to logits
 */
inline std::vector<std::vector<float>> sigmoid(const std::vector<std::vector<float>>& logits) {
    std::vector<std::vector<float>> result(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i].resize(logits[i].size());
        for (size_t j = 0; j < logits[i].size(); ++j) {
            result[i][j] = 1.0f / (1.0f + std::exp(-logits[i][j]));
        }
    }
    return result;
}

/**
 * @brief Apply softmax to logits
 */
inline std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& logits) {
    std::vector<std::vector<float>> result(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i].resize(logits[i].size());
        float max_val = *std::max_element(logits[i].begin(), logits[i].end());
        float sum = 0;
        for (size_t j = 0; j < logits[i].size(); ++j) {
            result[i][j] = std::exp(logits[i][j] - max_val);
            sum += result[i][j];
        }
        for (size_t j = 0; j < logits[i].size(); ++j) {
            result[i][j] /= sum;
        }
    }
    return result;
}

/**
 * @brief Filter detections by confidence threshold
 */
inline std::vector<Detection> filter_by_confidence(const std::vector<Detection>& detections,
                                                    float threshold) {
    std::vector<Detection> filtered;
    for (const auto& det : detections) {
        if (det.confidence >= threshold) {
            filtered.push_back(det);
        }
    }
    return filtered;
}

/**
 * @brief Rescale boxes from normalized coordinates
 */
inline void rescale_boxes(std::vector<Detection>& detections, int image_width, int image_height) {
    for (auto& det : detections) {
        det.bbox.x1 *= image_width;
        det.bbox.y1 *= image_height;
        det.bbox.x2 *= image_width;
        det.bbox.y2 *= image_height;
    }
}

/**
 * @brief Scale boxes from model input size to original image size
 */
inline void scale_boxes(std::vector<Detection>& detections,
                        int model_width, int model_height,
                        int orig_width, int orig_height,
                        bool letterbox = true) {
    float scale_x, scale_y;
    int offset_x = 0, offset_y = 0;
    
    if (letterbox) {
        float scale = std::min(static_cast<float>(model_width) / orig_width,
                               static_cast<float>(model_height) / orig_height);
        int new_w = static_cast<int>(orig_width * scale);
        int new_h = static_cast<int>(orig_height * scale);
        offset_x = (model_width - new_w) / 2;
        offset_y = (model_height - new_h) / 2;
        scale_x = scale_y = scale;
    } else {
        scale_x = static_cast<float>(model_width) / orig_width;
        scale_y = static_cast<float>(model_height) / orig_height;
    }
    
    for (auto& det : detections) {
        det.bbox.x1 = (det.bbox.x1 - offset_x) / scale_x;
        det.bbox.y1 = (det.bbox.y1 - offset_y) / scale_y;
        det.bbox.x2 = (det.bbox.x2 - offset_x) / scale_x;
        det.bbox.y2 = (det.bbox.y2 - offset_y) / scale_y;
        
        // Clip to image bounds
        det.bbox = det.bbox.clip(orig_width, orig_height);
    }
}

} // namespace postprocess

} // namespace object_detection
} // namespace neurova
