// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file object_detection.hpp
 * @brief Object Detection module main header
 * 
 * Neurova object detection framework including anchors, NMS, backbones, and detectors.
 */

#pragma once

// Core components
#include "object_detection/anchor.hpp"
#include "object_detection/nms.hpp"
#include "object_detection/backbone.hpp"
#include "object_detection/detector.hpp"

namespace neurova {
namespace object_detection {

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Run complete detection pipeline
 */
inline std::vector<Detection> detect(
    const std::vector<Detection>& raw_detections,
    float confidence_threshold = 0.5f,
    float nms_threshold = 0.4f) {
    
    // Filter by confidence
    auto filtered = postprocess::filter_by_confidence(raw_detections, confidence_threshold);
    
    // Apply NMS
    auto keep_indices = nms_per_class(filtered, nms_threshold);
    
    std::vector<Detection> results;
    results.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        results.push_back(filtered[idx]);
    }
    
    return results;
}

/**
 * @brief Create default COCO anchor configuration
 */
inline AnchorConfig coco_anchor_config() {
    AnchorConfig config;
    config.scales = {32.0f, 64.0f, 128.0f, 256.0f, 512.0f};
    config.ratios = {0.5f, 1.0f, 2.0f};
    config.feature_strides = {4, 8, 16, 32, 64};
    config.base_size = 4;
    return config;
}

/**
 * @brief Create default anchor configuration
 */
inline GridAnchors::Config default_anchor_config() {
    GridAnchors::Config config;
    config.anchors = {
        {{10, 13}, {16, 30}, {33, 23}},      // P3/8
        {{30, 61}, {62, 45}, {59, 119}},     // P4/16
        {{116, 90}, {156, 198}, {373, 326}}  // P5/32
    };
    config.strides = {8, 16, 32};
    config.input_size = 640;
    return config;
}

/**
 * @brief Create SSD300 default anchor configuration
 */
inline SSDAnchors::Config ssd300_anchor_config() {
    SSDAnchors::Config config;
    config.feature_map_sizes = {38, 19, 10, 5, 3, 1};
    config.min_sizes = {30.0f, 60.0f, 111.0f, 162.0f, 213.0f, 264.0f};
    config.max_sizes = {60.0f, 111.0f, 162.0f, 213.0f, 264.0f, 315.0f};
    config.aspect_ratios = {
        {2.0f},
        {2.0f, 3.0f},
        {2.0f, 3.0f},
        {2.0f, 3.0f},
        {2.0f},
        {2.0f}
    };
    config.image_size = 300;
    config.clip = true;
    return config;
}

// ============================================================================
// COCO Class Names
// ============================================================================

inline const std::vector<std::string>& coco_class_names() {
    static const std::vector<std::string> names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };
    return names;
}

/**
 * @brief Get class name from ID
 */
inline std::string get_class_name(int class_id, const std::vector<std::string>& names) {
    if (class_id >= 0 && class_id < static_cast<int>(names.size())) {
        return names[class_id];
    }
    return "unknown";
}

// ============================================================================
// Drawing Utilities
// ============================================================================

/**
 * @brief Draw detection box on image
 */
inline void draw_detection(std::vector<std::vector<std::vector<float>>>& image,
                           const Detection& det,
                           const std::array<float, 3>& color = {0, 255, 0},
                           int thickness = 2) {
    int height = image[0].size();
    int width = image[0][0].size();
    
    int x1 = static_cast<int>(std::max(0.0f, det.bbox.x1));
    int y1 = static_cast<int>(std::max(0.0f, det.bbox.y1));
    int x2 = static_cast<int>(std::min(static_cast<float>(width - 1), det.bbox.x2));
    int y2 = static_cast<int>(std::min(static_cast<float>(height - 1), det.bbox.y2));
    
    // Draw horizontal lines
    for (int t = 0; t < thickness; ++t) {
        for (int x = x1; x <= x2; ++x) {
            if (y1 + t < height) {
                for (int c = 0; c < 3 && c < static_cast<int>(image.size()); ++c) {
                    image[c][y1 + t][x] = color[c];
                }
            }
            if (y2 - t >= 0) {
                for (int c = 0; c < 3 && c < static_cast<int>(image.size()); ++c) {
                    image[c][y2 - t][x] = color[c];
                }
            }
        }
    }
    
    // Draw vertical lines
    for (int t = 0; t < thickness; ++t) {
        for (int y = y1; y <= y2; ++y) {
            if (x1 + t < width) {
                for (int c = 0; c < 3 && c < static_cast<int>(image.size()); ++c) {
                    image[c][y][x1 + t] = color[c];
                }
            }
            if (x2 - t >= 0) {
                for (int c = 0; c < 3 && c < static_cast<int>(image.size()); ++c) {
                    image[c][y][x2 - t] = color[c];
                }
            }
        }
    }
}

/**
 * @brief Draw all detections on image
 */
inline void draw_detections(std::vector<std::vector<std::vector<float>>>& image,
                            const std::vector<Detection>& detections,
                            int thickness = 2) {
    // Color palette (RGB)
    static const std::vector<std::array<float, 3>> colors = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255},
        {0, 255, 255}, {128, 0, 0}, {0, 128, 0}, {0, 0, 128}, {128, 128, 0},
        {128, 0, 128}, {0, 128, 128}, {255, 128, 0}, {255, 0, 128}, {128, 255, 0}
    };
    
    for (const auto& det : detections) {
        auto color = colors[det.class_id % colors.size()];
        draw_detection(image, det, color, thickness);
    }
}

// ============================================================================
// Evaluation Metrics
// ============================================================================

/**
 * @brief Calculate precision and recall
 */
inline std::pair<float, float> precision_recall(
    const std::vector<Detection>& predictions,
    const std::vector<Detection>& ground_truth,
    float iou_threshold = 0.5f) {
    
    if (predictions.empty() || ground_truth.empty()) {
        return {0.0f, 0.0f};
    }
    
    std::vector<bool> gt_matched(ground_truth.size(), false);
    int tp = 0;
    
    for (const auto& pred : predictions) {
        float best_iou = 0;
        int best_gt = -1;
        
        for (size_t i = 0; i < ground_truth.size(); ++i) {
            if (gt_matched[i]) continue;
            if (pred.class_id != ground_truth[i].class_id) continue;
            
            float iou_val = iou(pred.bbox, ground_truth[i].bbox);
            if (iou_val > best_iou) {
                best_iou = iou_val;
                best_gt = static_cast<int>(i);
            }
        }
        
        if (best_iou >= iou_threshold && best_gt >= 0) {
            tp++;
            gt_matched[best_gt] = true;
        }
    }
    
    float precision = static_cast<float>(tp) / predictions.size();
    float recall = static_cast<float>(tp) / ground_truth.size();
    
    return {precision, recall};
}

/**
 * @brief Calculate Average Precision (AP)
 */
inline float average_precision(
    const std::vector<Detection>& predictions,
    const std::vector<Detection>& ground_truth,
    float iou_threshold = 0.5f) {
    
    if (ground_truth.empty()) return 0.0f;
    
    // Sort predictions by confidence
    auto sorted_preds = predictions;
    std::sort(sorted_preds.begin(), sorted_preds.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<bool> gt_matched(ground_truth.size(), false);
    std::vector<float> precisions, recalls;
    
    int tp = 0;
    int fp = 0;
    
    for (const auto& pred : sorted_preds) {
        float best_iou = 0;
        int best_gt = -1;
        
        for (size_t i = 0; i < ground_truth.size(); ++i) {
            if (gt_matched[i]) continue;
            if (pred.class_id != ground_truth[i].class_id) continue;
            
            float iou_val = iou(pred.bbox, ground_truth[i].bbox);
            if (iou_val > best_iou) {
                best_iou = iou_val;
                best_gt = static_cast<int>(i);
            }
        }
        
        if (best_iou >= iou_threshold && best_gt >= 0) {
            tp++;
            gt_matched[best_gt] = true;
        } else {
            fp++;
        }
        
        precisions.push_back(static_cast<float>(tp) / (tp + fp));
        recalls.push_back(static_cast<float>(tp) / ground_truth.size());
    }
    
    // Calculate AP using 11-point interpolation
    float ap = 0;
    for (float t = 0; t <= 1.0f; t += 0.1f) {
        float max_prec = 0;
        for (size_t i = 0; i < recalls.size(); ++i) {
            if (recalls[i] >= t) {
                max_prec = std::max(max_prec, precisions[i]);
            }
        }
        ap += max_prec / 11.0f;
    }
    
    return ap;
}

/**
 * @brief Calculate mean Average Precision (mAP)
 */
inline float mean_average_precision(
    const std::vector<std::vector<Detection>>& all_predictions,
    const std::vector<std::vector<Detection>>& all_ground_truth,
    float iou_threshold = 0.5f,
    int num_classes = 80) {
    
    // Collect per-class predictions and ground truth
    std::vector<std::vector<Detection>> class_preds(num_classes);
    std::vector<std::vector<Detection>> class_gt(num_classes);
    
    for (size_t img = 0; img < all_predictions.size(); ++img) {
        for (const auto& pred : all_predictions[img]) {
            if (pred.class_id >= 0 && pred.class_id < num_classes) {
                class_preds[pred.class_id].push_back(pred);
            }
        }
        if (img < all_ground_truth.size()) {
            for (const auto& gt : all_ground_truth[img]) {
                if (gt.class_id >= 0 && gt.class_id < num_classes) {
                    class_gt[gt.class_id].push_back(gt);
                }
            }
        }
    }
    
    // Calculate AP for each class
    float sum_ap = 0;
    int valid_classes = 0;
    
    for (int c = 0; c < num_classes; ++c) {
        if (!class_gt[c].empty()) {
            sum_ap += average_precision(class_preds[c], class_gt[c], iou_threshold);
            valid_classes++;
        }
    }
    
    return valid_classes > 0 ? sum_ap / valid_classes : 0.0f;
}

} // namespace object_detection
} // namespace neurova
