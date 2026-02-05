// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file object_detection/anchor.hpp
 * @brief Anchor box generation for object detection
 * 
 * Neurova implementation of anchor boxes used in modern object detectors.
 */

#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

namespace neurova {
namespace object_detection {

/**
 * @brief Bounding box representation
 */
struct BBox {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;  // (x1, y1) top-left, (x2, y2) bottom-right
    
    BBox() = default;
    BBox(float x1_, float y1_, float x2_, float y2_) : x1(x1_), y1(y1_), x2(x2_), y2(y2_) {}
    
    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area() const { return width() * height(); }
    float center_x() const { return (x1 + x2) / 2; }
    float center_y() const { return (y1 + y2) / 2; }
    
    // Convert to (cx, cy, w, h) format
    std::array<float, 4> to_cxcywh() const {
        return {center_x(), center_y(), width(), height()};
    }
    
    // Create from (cx, cy, w, h) format
    static BBox from_cxcywh(float cx, float cy, float w, float h) {
        return BBox(cx - w/2, cy - h/2, cx + w/2, cy + h/2);
    }
    
    // Clip to image boundaries
    BBox clip(int img_width, int img_height) const {
        return BBox(
            std::max(0.0f, std::min(x1, static_cast<float>(img_width))),
            std::max(0.0f, std::min(y1, static_cast<float>(img_height))),
            std::max(0.0f, std::min(x2, static_cast<float>(img_width))),
            std::max(0.0f, std::min(y2, static_cast<float>(img_height)))
        );
    }
};

/**
 * @brief Detection result with confidence and class
 */
struct Detection {
    BBox bbox;
    float confidence = 0;
    int class_id = -1;
    
    Detection() = default;
    Detection(const BBox& b, float conf, int cls) : bbox(b), confidence(conf), class_id(cls) {}
};

/**
 * @brief Calculate IoU (Intersection over Union) between two boxes
 */
inline float iou(const BBox& a, const BBox& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    
    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
    float union_area = a.area() + b.area() - inter_area;
    
    return union_area > 0 ? inter_area / union_area : 0;
}

/**
 * @brief Calculate GIoU (Generalized IoU)
 */
inline float giou(const BBox& a, const BBox& b) {
    float iou_val = iou(a, b);
    
    // Enclosing box
    float enc_x1 = std::min(a.x1, b.x1);
    float enc_y1 = std::min(a.y1, b.y1);
    float enc_x2 = std::max(a.x2, b.x2);
    float enc_y2 = std::max(a.y2, b.y2);
    
    float enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1);
    float union_area = a.area() + b.area() - iou_val * (a.area() + b.area() - iou_val);
    
    return iou_val - (enc_area - union_area) / enc_area;
}

/**
 * @brief Calculate DIoU (Distance IoU)
 */
inline float diou(const BBox& a, const BBox& b) {
    float iou_val = iou(a, b);
    
    // Center distance
    float center_dist_sq = std::pow(a.center_x() - b.center_x(), 2) + 
                          std::pow(a.center_y() - b.center_y(), 2);
    
    // Diagonal of enclosing box
    float enc_x1 = std::min(a.x1, b.x1);
    float enc_y1 = std::min(a.y1, b.y1);
    float enc_x2 = std::max(a.x2, b.x2);
    float enc_y2 = std::max(a.y2, b.y2);
    float enc_diag_sq = std::pow(enc_x2 - enc_x1, 2) + std::pow(enc_y2 - enc_y1, 2);
    
    return iou_val - center_dist_sq / (enc_diag_sq + 1e-6f);
}

/**
 * @brief Calculate CIoU (Complete IoU)
 */
inline float ciou(const BBox& a, const BBox& b) {
    float iou_val = iou(a, b);
    
    // Center distance
    float center_dist_sq = std::pow(a.center_x() - b.center_x(), 2) + 
                          std::pow(a.center_y() - b.center_y(), 2);
    
    // Diagonal of enclosing box
    float enc_x1 = std::min(a.x1, b.x1);
    float enc_y1 = std::min(a.y1, b.y1);
    float enc_x2 = std::max(a.x2, b.x2);
    float enc_y2 = std::max(a.y2, b.y2);
    float enc_diag_sq = std::pow(enc_x2 - enc_x1, 2) + std::pow(enc_y2 - enc_y1, 2);
    
    // Aspect ratio consistency
    constexpr float pi = 3.14159265358979f;
    float v = (4.0f / (pi * pi)) * std::pow(
        std::atan(b.width() / (b.height() + 1e-6f)) - 
        std::atan(a.width() / (a.height() + 1e-6f)), 2);
    
    float alpha = v / (1 - iou_val + v + 1e-6f);
    
    return iou_val - center_dist_sq / (enc_diag_sq + 1e-6f) - alpha * v;
}

/**
 * @brief Anchor generator configuration
 */
struct AnchorConfig {
    std::vector<float> scales = {8.0f, 16.0f, 32.0f};          // Anchor scales
    std::vector<float> ratios = {0.5f, 1.0f, 2.0f};            // Aspect ratios
    std::vector<int> feature_strides = {8, 16, 32};            // Feature map strides
    int base_size = 16;                                         // Base anchor size
};

/**
 * @brief Generate anchor boxes for a feature map
 */
inline std::vector<BBox> generate_anchors(int feature_width, int feature_height,
                                          int stride, int base_size,
                                          const std::vector<float>& scales,
                                          const std::vector<float>& ratios) {
    std::vector<BBox> anchors;
    
    // Generate base anchors
    std::vector<BBox> base_anchors;
    for (float scale : scales) {
        for (float ratio : ratios) {
            float w = base_size * scale * std::sqrt(ratio);
            float h = base_size * scale / std::sqrt(ratio);
            base_anchors.push_back(BBox::from_cxcywh(0, 0, w, h));
        }
    }
    
    // Tile anchors over feature map
    for (int y = 0; y < feature_height; ++y) {
        for (int x = 0; x < feature_width; ++x) {
            float cx = (x + 0.5f) * stride;
            float cy = (y + 0.5f) * stride;
            
            for (const auto& base : base_anchors) {
                anchors.push_back(BBox(
                    cx + base.x1, cy + base.y1,
                    cx + base.x2, cy + base.y2
                ));
            }
        }
    }
    
    return anchors;
}

/**
 * @brief Generate multi-scale anchors for FPN-style networks
 */
inline std::vector<std::vector<BBox>> generate_multiscale_anchors(
    int image_width, int image_height,
    const AnchorConfig& config) {
    
    std::vector<std::vector<BBox>> all_anchors;
    
    for (size_t level = 0; level < config.feature_strides.size(); ++level) {
        int stride = config.feature_strides[level];
        int feat_w = (image_width + stride - 1) / stride;
        int feat_h = (image_height + stride - 1) / stride;
        
        // Scale for this level
        std::vector<float> level_scales = {config.scales[level]};
        
        auto anchors = generate_anchors(feat_w, feat_h, stride, config.base_size,
                                        level_scales, config.ratios);
        all_anchors.push_back(std::move(anchors));
    }
    
    return all_anchors;
}

/**
 * @brief SSD-style anchor generator
 */
class SSDAnchors {
public:
    struct Config {
        std::vector<int> feature_map_sizes;    // e.g., {38, 19, 10, 5, 3, 1}
        std::vector<float> min_sizes;          // Min anchor sizes per level
        std::vector<float> max_sizes;          // Max anchor sizes per level
        std::vector<std::vector<float>> aspect_ratios;  // Aspect ratios per level
        int image_size = 300;
        bool clip = true;
    };
    
    explicit SSDAnchors(const Config& config) : config_(config) {}
    
    std::vector<BBox> generate() const {
        std::vector<BBox> anchors;
        
        for (size_t k = 0; k < config_.feature_map_sizes.size(); ++k) {
            int f_k = config_.feature_map_sizes[k];
            float s_k = config_.min_sizes[k] / config_.image_size;
            float s_k_prime = std::sqrt(s_k * config_.max_sizes[k] / config_.image_size);
            
            for (int i = 0; i < f_k; ++i) {
                for (int j = 0; j < f_k; ++j) {
                    float cx = (j + 0.5f) / f_k;
                    float cy = (i + 0.5f) / f_k;
                    
                    // Anchor with aspect ratio 1
                    anchors.push_back(make_anchor(cx, cy, s_k, s_k));
                    anchors.push_back(make_anchor(cx, cy, s_k_prime, s_k_prime));
                    
                    // Additional aspect ratios
                    for (float ar : config_.aspect_ratios[k]) {
                        if (std::abs(ar - 1.0f) > 1e-6f) {
                            float w = s_k * std::sqrt(ar);
                            float h = s_k / std::sqrt(ar);
                            anchors.push_back(make_anchor(cx, cy, w, h));
                            anchors.push_back(make_anchor(cx, cy, h, w));
                        }
                    }
                }
            }
        }
        
        if (config_.clip) {
            for (auto& anchor : anchors) {
                anchor.x1 = std::max(0.0f, std::min(1.0f, anchor.x1));
                anchor.y1 = std::max(0.0f, std::min(1.0f, anchor.y1));
                anchor.x2 = std::max(0.0f, std::min(1.0f, anchor.x2));
                anchor.y2 = std::max(0.0f, std::min(1.0f, anchor.y2));
            }
        }
        
        return anchors;
    }
    
private:
    BBox make_anchor(float cx, float cy, float w, float h) const {
        return BBox(cx - w/2, cy - h/2, cx + w/2, cy + h/2);
    }
    
    Config config_;
};

/**
 * @brief Grid-based anchor generator
 */
class GridAnchors {
public:
    struct Config {
        std::vector<std::vector<std::pair<int, int>>> anchors;  // Anchors per scale
        std::vector<int> strides = {8, 16, 32};                  // Stride per scale
        int input_size = 640;
    };
    
    explicit GridAnchors(const Config& config) : config_(config) {}
    
    std::vector<std::vector<BBox>> generate(int image_width, int image_height) const {
        std::vector<std::vector<BBox>> all_anchors;
        
        for (size_t scale = 0; scale < config_.strides.size(); ++scale) {
            int stride = config_.strides[scale];
            int grid_w = image_width / stride;
            int grid_h = image_height / stride;
            
            std::vector<BBox> level_anchors;
            
            for (int gy = 0; gy < grid_h; ++gy) {
                for (int gx = 0; gx < grid_w; ++gx) {
                    float cx = (gx + 0.5f) * stride;
                    float cy = (gy + 0.5f) * stride;
                    
                    for (const auto& [aw, ah] : config_.anchors[scale]) {
                        level_anchors.push_back(BBox::from_cxcywh(cx, cy, 
                            static_cast<float>(aw), static_cast<float>(ah)));
                    }
                }
            }
            
            all_anchors.push_back(std::move(level_anchors));
        }
        
        return all_anchors;
    }
    
private:
    Config config_;
};

/**
 * @brief Encode box deltas (target -> anchor delta)
 */
inline std::array<float, 4> encode_box(const BBox& target, const BBox& anchor,
                                        float weight_x = 1.0f, float weight_y = 1.0f,
                                        float weight_w = 1.0f, float weight_h = 1.0f) {
    float anchor_w = anchor.width();
    float anchor_h = anchor.height();
    
    float tx = (target.center_x() - anchor.center_x()) / anchor_w / weight_x;
    float ty = (target.center_y() - anchor.center_y()) / anchor_h / weight_y;
    float tw = std::log(target.width() / anchor_w) / weight_w;
    float th = std::log(target.height() / anchor_h) / weight_h;
    
    return {tx, ty, tw, th};
}

/**
 * @brief Decode box deltas (anchor + delta -> box)
 */
inline BBox decode_box(const std::array<float, 4>& delta, const BBox& anchor,
                       float weight_x = 1.0f, float weight_y = 1.0f,
                       float weight_w = 1.0f, float weight_h = 1.0f) {
    float anchor_w = anchor.width();
    float anchor_h = anchor.height();
    
    float cx = delta[0] * weight_x * anchor_w + anchor.center_x();
    float cy = delta[1] * weight_y * anchor_h + anchor.center_y();
    float w = std::exp(delta[2] * weight_w) * anchor_w;
    float h = std::exp(delta[3] * weight_h) * anchor_h;
    
    return BBox::from_cxcywh(cx, cy, w, h);
}

/**
 * @brief Match anchors to ground truth boxes
 */
inline std::vector<int> match_anchors(const std::vector<BBox>& anchors,
                                       const std::vector<BBox>& gt_boxes,
                                       float pos_threshold = 0.5f,
                                       float neg_threshold = 0.4f) {
    std::vector<int> matches(anchors.size(), -1);  // -1 = negative, >=0 = matched gt index
    
    if (gt_boxes.empty()) return matches;
    
    // Calculate IoU matrix
    for (size_t i = 0; i < anchors.size(); ++i) {
        float best_iou = 0;
        int best_gt = -1;
        
        for (size_t j = 0; j < gt_boxes.size(); ++j) {
            float iou_val = iou(anchors[i], gt_boxes[j]);
            if (iou_val > best_iou) {
                best_iou = iou_val;
                best_gt = static_cast<int>(j);
            }
        }
        
        if (best_iou >= pos_threshold) {
            matches[i] = best_gt;
        } else if (best_iou < neg_threshold) {
            matches[i] = -1;  // Negative
        } else {
            matches[i] = -2;  // Ignore
        }
    }
    
    // Ensure each GT has at least one positive anchor
    for (size_t j = 0; j < gt_boxes.size(); ++j) {
        float best_iou = 0;
        size_t best_anchor = 0;
        
        for (size_t i = 0; i < anchors.size(); ++i) {
            float iou_val = iou(anchors[i], gt_boxes[j]);
            if (iou_val > best_iou) {
                best_iou = iou_val;
                best_anchor = i;
            }
        }
        
        matches[best_anchor] = static_cast<int>(j);
    }
    
    return matches;
}

} // namespace object_detection
} // namespace neurova
