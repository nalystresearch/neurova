// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file object_detection/nms.hpp
 * @brief Non-Maximum Suppression algorithms
 * 
 * Neurova implementation of various NMS algorithms for object detection.
 */

#pragma once

#include "anchor.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace neurova {
namespace object_detection {

/**
 * @brief Standard Non-Maximum Suppression
 * 
 * @param detections Input detections
 * @param iou_threshold IoU threshold for suppression
 * @return Indices of kept detections
 */
inline std::vector<int> nms(const std::vector<Detection>& detections,
                            float iou_threshold = 0.5f) {
    if (detections.empty()) return {};
    
    // Sort by confidence (descending)
    std::vector<int> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });
    
    std::vector<int> keep;
    std::vector<bool> suppressed(detections.size(), false);
    
    for (int idx : indices) {
        if (suppressed[idx]) continue;
        
        keep.push_back(idx);
        
        for (int other : indices) {
            if (suppressed[other] || other == idx) continue;
            
            float iou_val = iou(detections[idx].bbox, detections[other].bbox);
            if (iou_val > iou_threshold) {
                suppressed[other] = true;
            }
        }
    }
    
    return keep;
}

/**
 * @brief Class-aware NMS (separate NMS per class)
 */
inline std::vector<int> nms_per_class(const std::vector<Detection>& detections,
                                       float iou_threshold = 0.5f) {
    if (detections.empty()) return {};
    
    // Group by class
    std::map<int, std::vector<int>> class_indices;
    for (size_t i = 0; i < detections.size(); ++i) {
        class_indices[detections[i].class_id].push_back(static_cast<int>(i));
    }
    
    std::vector<int> all_keep;
    
    for (auto& [cls, indices] : class_indices) {
        // Create subset
        std::vector<Detection> subset;
        for (int idx : indices) {
            subset.push_back(detections[idx]);
        }
        
        // Run NMS on subset
        auto keep = nms(subset, iou_threshold);
        
        // Map back to original indices
        for (int k : keep) {
            all_keep.push_back(indices[k]);
        }
    }
    
    return all_keep;
}

/**
 * @brief Soft-NMS (Bodla et al., 2017)
 * 
 * Instead of hard suppression, reduces confidence of overlapping boxes
 */
enum class SoftNMSMethod {
    Linear,
    Gaussian
};

inline std::vector<Detection> soft_nms(std::vector<Detection> detections,
                                        float iou_threshold = 0.3f,
                                        float score_threshold = 0.001f,
                                        float sigma = 0.5f,
                                        SoftNMSMethod method = SoftNMSMethod::Gaussian) {
    if (detections.empty()) return {};
    
    std::vector<Detection> result;
    
    while (!detections.empty()) {
        // Find max confidence
        auto max_it = std::max_element(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) {
                return a.confidence < b.confidence;
            });
        
        if (max_it->confidence < score_threshold) break;
        
        Detection max_det = *max_it;
        result.push_back(max_det);
        detections.erase(max_it);
        
        // Update confidences
        for (auto& det : detections) {
            float iou_val = iou(max_det.bbox, det.bbox);
            
            if (method == SoftNMSMethod::Linear) {
                if (iou_val > iou_threshold) {
                    det.confidence *= (1 - iou_val);
                }
            } else {  // Gaussian
                det.confidence *= std::exp(-(iou_val * iou_val) / sigma);
            }
        }
        
        // Remove low confidence
        detections.erase(
            std::remove_if(detections.begin(), detections.end(),
                [score_threshold](const Detection& d) {
                    return d.confidence < score_threshold;
                }),
            detections.end());
    }
    
    return result;
}

/**
 * @brief DIoU-NMS (Zheng et al., 2019)
 * 
 * Uses Distance-IoU instead of standard IoU
 */
inline std::vector<int> diou_nms(const std::vector<Detection>& detections,
                                  float iou_threshold = 0.5f) {
    if (detections.empty()) return {};
    
    std::vector<int> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });
    
    std::vector<int> keep;
    std::vector<bool> suppressed(detections.size(), false);
    
    for (int idx : indices) {
        if (suppressed[idx]) continue;
        
        keep.push_back(idx);
        
        for (int other : indices) {
            if (suppressed[other] || other == idx) continue;
            
            float diou_val = diou(detections[idx].bbox, detections[other].bbox);
            if (diou_val > iou_threshold) {
                suppressed[other] = true;
            }
        }
    }
    
    return keep;
}

/**
 * @brief Weighted NMS (Zhou et al., 2017)
 * 
 * Merges overlapping boxes instead of suppressing them
 */
inline std::vector<Detection> weighted_nms(const std::vector<Detection>& detections,
                                            float iou_threshold = 0.5f) {
    if (detections.empty()) return {};
    
    std::vector<Detection> input = detections;
    std::vector<Detection> result;
    
    // Sort by confidence
    std::sort(input.begin(), input.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    
    std::vector<bool> merged(input.size(), false);
    
    for (size_t i = 0; i < input.size(); ++i) {
        if (merged[i]) continue;
        
        // Find all boxes to merge
        std::vector<size_t> to_merge = {i};
        for (size_t j = i + 1; j < input.size(); ++j) {
            if (merged[j]) continue;
            if (input[i].class_id != input[j].class_id) continue;
            
            if (iou(input[i].bbox, input[j].bbox) > iou_threshold) {
                to_merge.push_back(j);
                merged[j] = true;
            }
        }
        
        // Weighted merge
        float sum_conf = 0;
        float sum_x1 = 0, sum_y1 = 0, sum_x2 = 0, sum_y2 = 0;
        
        for (size_t idx : to_merge) {
            float w = input[idx].confidence;
            sum_conf += w;
            sum_x1 += w * input[idx].bbox.x1;
            sum_y1 += w * input[idx].bbox.y1;
            sum_x2 += w * input[idx].bbox.x2;
            sum_y2 += w * input[idx].bbox.y2;
        }
        
        Detection merged_det;
        merged_det.bbox = BBox(sum_x1 / sum_conf, sum_y1 / sum_conf,
                               sum_x2 / sum_conf, sum_y2 / sum_conf);
        merged_det.confidence = input[i].confidence;  // Keep max confidence
        merged_det.class_id = input[i].class_id;
        
        result.push_back(merged_det);
    }
    
    return result;
}

/**
 * @brief Matrix NMS (Wang et al., SOLOv2)
 * 
 * Efficient parallel NMS using matrix operations
 */
inline std::vector<Detection> matrix_nms(std::vector<Detection> detections,
                                          float sigma = 2.0f,
                                          float score_threshold = 0.01f,
                                          int max_detections = 500) {
    if (detections.empty()) return {};
    
    // Sort by confidence and limit
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    
    if (detections.size() > static_cast<size_t>(max_detections)) {
        detections.resize(max_detections);
    }
    
    size_t n = detections.size();
    
    // Compute IoU matrix
    std::vector<std::vector<float>> iou_matrix(n, std::vector<float>(n, 0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            float iou_val = iou(detections[i].bbox, detections[j].bbox);
            iou_matrix[i][j] = iou_val;
            iou_matrix[j][i] = iou_val;
        }
    }
    
    // Compute decay factors
    std::vector<float> scores(n);
    for (size_t i = 0; i < n; ++i) {
        scores[i] = detections[i].confidence;
    }
    
    for (size_t i = 0; i < n; ++i) {
        float max_iou = 0;
        for (size_t j = 0; j < i; ++j) {
            if (detections[i].class_id == detections[j].class_id) {
                max_iou = std::max(max_iou, iou_matrix[i][j]);
            }
        }
        
        float decay = std::exp(-(max_iou * max_iou) / sigma);
        scores[i] *= decay;
    }
    
    // Filter by threshold
    std::vector<Detection> result;
    for (size_t i = 0; i < n; ++i) {
        if (scores[i] >= score_threshold) {
            Detection det = detections[i];
            det.confidence = scores[i];
            result.push_back(det);
        }
    }
    
    return result;
}

/**
 * @brief Batched NMS for multiple images
 */
inline std::vector<std::vector<int>> batched_nms(
    const std::vector<std::vector<Detection>>& batch_detections,
    float iou_threshold = 0.5f) {
    
    std::vector<std::vector<int>> batch_keep;
    batch_keep.reserve(batch_detections.size());
    
    for (const auto& detections : batch_detections) {
        batch_keep.push_back(nms(detections, iou_threshold));
    }
    
    return batch_keep;
}

/**
 * @brief Fast NMS (Bolya et al., YOLACT)
 */
inline std::vector<int> fast_nms(const std::vector<Detection>& detections,
                                  float iou_threshold = 0.5f,
                                  int top_k = 200) {
    if (detections.empty()) return {};
    
    // Sort and take top-k
    std::vector<int> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });
    
    if (indices.size() > static_cast<size_t>(top_k)) {
        indices.resize(top_k);
    }
    
    // Compute IoU matrix for top-k
    size_t n = indices.size();
    std::vector<std::vector<float>> iou_matrix(n, std::vector<float>(n, 0));
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            iou_matrix[i][j] = iou(detections[indices[i]].bbox, 
                                   detections[indices[j]].bbox);
        }
    }
    
    // Keep boxes where max IoU with higher-scoring boxes is below threshold
    std::vector<int> keep;
    for (size_t i = 0; i < n; ++i) {
        float max_iou = 0;
        for (size_t j = 0; j < i; ++j) {
            if (detections[indices[i]].class_id == detections[indices[j]].class_id) {
                max_iou = std::max(max_iou, iou_matrix[j][i]);
            }
        }
        
        if (max_iou <= iou_threshold) {
            keep.push_back(indices[i]);
        }
    }
    
    return keep;
}

/**
 * @brief Cluster NMS
 * 
 * Groups detections into clusters and selects representative from each
 */
inline std::vector<int> cluster_nms(const std::vector<Detection>& detections,
                                     float iou_threshold = 0.5f,
                                     int max_iterations = 100) {
    if (detections.empty()) return {};
    
    std::vector<int> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });
    
    size_t n = indices.size();
    std::vector<bool> suppressed(n, false);
    
    // Iteratively suppress until convergence
    for (int iter = 0; iter < max_iterations; ++iter) {
        bool changed = false;
        
        for (size_t i = 0; i < n; ++i) {
            if (suppressed[i]) continue;
            
            for (size_t j = i + 1; j < n; ++j) {
                if (suppressed[j]) continue;
                if (detections[indices[i]].class_id != detections[indices[j]].class_id) continue;
                
                float iou_val = iou(detections[indices[i]].bbox, detections[indices[j]].bbox);
                if (iou_val > iou_threshold) {
                    suppressed[j] = true;
                    changed = true;
                }
            }
        }
        
        if (!changed) break;
    }
    
    std::vector<int> keep;
    for (size_t i = 0; i < n; ++i) {
        if (!suppressed[i]) {
            keep.push_back(indices[i]);
        }
    }
    
    return keep;
}

} // namespace object_detection
} // namespace neurova
