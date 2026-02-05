// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

#ifndef NEUROVA_DETECTION_TEMPLATE_MATCHING_HPP
#define NEUROVA_DETECTION_TEMPLATE_MATCHING_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace neurova {
namespace detection {

/**
 * @brief Template matching method enumeration
 */
enum class TemplateMethod {
    NCC,    // Normalized cross-correlation (range [-1, 1])
    SSD,    // Sum of squared differences (lower is better)
    SAD,    // Sum of absolute differences (lower is better)
    CCORR   // Cross-correlation (unnormalized)
};

/**
 * @brief Bounding box with confidence score
 */
struct Detection {
    int x, y, width, height;
    float confidence;
    
    Detection(int x_ = 0, int y_ = 0, int w_ = 0, int h_ = 0, float conf_ = 0.0f)
        : x(x_), y(y_), width(w_), height(h_), confidence(conf_) {}
};

/**
 * @brief Match template in image using various correlation methods
 * 
 * @param image Input image data (grayscale, row-major)
 * @param img_width Image width
 * @param img_height Image height
 * @param templ Template data (grayscale, row-major)
 * @param templ_width Template width
 * @param templ_height Template height
 * @param result Output response map
 * @param method Matching method
 */
inline void match_template(
    const float* image, int img_width, int img_height,
    const float* templ, int templ_width, int templ_height,
    float* result,
    TemplateMethod method = TemplateMethod::NCC
) {
    const int th = templ_height;
    const int tw = templ_width;
    const int ih = img_height;
    const int iw = img_width;
    
    if (th > ih || tw > iw) {
        throw std::invalid_argument("Template must be smaller than image");
    }
    
    // Initialize result
    for (int i = 0; i < ih * iw; ++i) {
        result[i] = (method == TemplateMethod::SSD || method == TemplateMethod::SAD) 
                    ? std::numeric_limits<float>::infinity() : 0.0f;
    }
    
    // Precompute template statistics for NCC
    float templ_mean = 0.0f, templ_std = 0.0f;
    if (method == TemplateMethod::NCC) {
        for (int i = 0; i < th * tw; ++i) {
            templ_mean += templ[i];
        }
        templ_mean /= (th * tw);
        
        for (int i = 0; i < th * tw; ++i) {
            float diff = templ[i] - templ_mean;
            templ_std += diff * diff;
        }
        templ_std = std::sqrt(templ_std / (th * tw));
        if (templ_std < 1e-6f) templ_std = 1.0f;
    }
    
    // Slide template over image
    for (int i = 0; i <= ih - th; ++i) {
        for (int j = 0; j <= iw - tw; ++j) {
            float value = 0.0f;
            
            switch (method) {
                case TemplateMethod::NCC: {
                    // Compute patch statistics
                    float patch_mean = 0.0f, patch_std = 0.0f;
                    for (int pi = 0; pi < th; ++pi) {
                        for (int pj = 0; pj < tw; ++pj) {
                            patch_mean += image[(i + pi) * iw + (j + pj)];
                        }
                    }
                    patch_mean /= (th * tw);
                    
                    for (int pi = 0; pi < th; ++pi) {
                        for (int pj = 0; pj < tw; ++pj) {
                            float diff = image[(i + pi) * iw + (j + pj)] - patch_mean;
                            patch_std += diff * diff;
                        }
                    }
                    patch_std = std::sqrt(patch_std / (th * tw));
                    if (patch_std < 1e-6f) patch_std = 1.0f;
                    
                    // NCC
                    float ncc = 0.0f;
                    for (int pi = 0; pi < th; ++pi) {
                        for (int pj = 0; pj < tw; ++pj) {
                            float patch_norm = (image[(i + pi) * iw + (j + pj)] - patch_mean) / patch_std;
                            float templ_norm = (templ[pi * tw + pj] - templ_mean) / templ_std;
                            ncc += patch_norm * templ_norm;
                        }
                    }
                    value = ncc / (th * tw);
                    break;
                }
                case TemplateMethod::SSD: {
                    for (int pi = 0; pi < th; ++pi) {
                        for (int pj = 0; pj < tw; ++pj) {
                            float diff = image[(i + pi) * iw + (j + pj)] - templ[pi * tw + pj];
                            value += diff * diff;
                        }
                    }
                    break;
                }
                case TemplateMethod::SAD: {
                    for (int pi = 0; pi < th; ++pi) {
                        for (int pj = 0; pj < tw; ++pj) {
                            value += std::abs(image[(i + pi) * iw + (j + pj)] - templ[pi * tw + pj]);
                        }
                    }
                    break;
                }
                case TemplateMethod::CCORR: {
                    for (int pi = 0; pi < th; ++pi) {
                        for (int pj = 0; pj < tw; ++pj) {
                            value += image[(i + pi) * iw + (j + pj)] * templ[pi * tw + pj];
                        }
                    }
                    break;
                }
            }
            
            result[(i + th/2) * iw + (j + tw/2)] = value;
        }
    }
}

/**
 * @brief Non-maximum suppression for bounding boxes
 * 
 * @param boxes Input detections
 * @param iou_threshold IoU threshold for suppression
 * @return Indices of kept detections
 */
inline std::vector<int> non_max_suppression(
    const std::vector<Detection>& boxes,
    float iou_threshold = 0.5f
) {
    if (boxes.empty()) return {};
    
    // Sort by confidence descending
    std::vector<int> indices(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) indices[i] = static_cast<int>(i);
    
    std::sort(indices.begin(), indices.end(), [&boxes](int a, int b) {
        return boxes[a].confidence > boxes[b].confidence;
    });
    
    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);
    
    for (int idx : indices) {
        if (suppressed[idx]) continue;
        
        keep.push_back(idx);
        const Detection& a = boxes[idx];
        
        for (int other : indices) {
            if (suppressed[other] || other == idx) continue;
            
            const Detection& b = boxes[other];
            
            // Compute IoU
            int x1 = std::max(a.x, b.x);
            int y1 = std::max(a.y, b.y);
            int x2 = std::min(a.x + a.width, b.x + b.width);
            int y2 = std::min(a.y + a.height, b.y + b.height);
            
            if (x2 <= x1 || y2 <= y1) continue;
            
            float intersection = static_cast<float>((x2 - x1) * (y2 - y1));
            float union_area = static_cast<float>(a.width * a.height + b.width * b.height) - intersection;
            float iou = intersection / std::max(union_area, 1e-6f);
            
            if (iou > iou_threshold) {
                suppressed[other] = true;
            }
        }
    }
    
    return keep;
}

/**
 * @brief Template-based object detector
 */
class TemplateDetector {
public:
    TemplateDetector(
        const float* templ, int templ_width, int templ_height,
        float threshold = 0.7f,
        TemplateMethod method = TemplateMethod::NCC,
        float nms_threshold = 0.3f
    ) : threshold_(threshold), method_(method), nms_threshold_(nms_threshold),
        templ_width_(templ_width), templ_height_(templ_height) {
        template_.assign(templ, templ + templ_width * templ_height);
    }
    
    /**
     * @brief Detect template in image
     * 
     * @param image Input image (grayscale)
     * @param img_width Image width
     * @param img_height Image height
     * @return Vector of detections
     */
    std::vector<Detection> detect(const float* image, int img_width, int img_height) {
        std::vector<float> response(img_width * img_height);
        match_template(image, img_width, img_height,
                      template_.data(), templ_width_, templ_height_,
                      response.data(), method_);
        
        std::vector<Detection> candidates;
        
        for (int i = 0; i < img_height; ++i) {
            for (int j = 0; j < img_width; ++j) {
                float score = response[i * img_width + j];
                bool is_detection = false;
                
                if (method_ == TemplateMethod::NCC || method_ == TemplateMethod::CCORR) {
                    is_detection = score >= threshold_;
                } else {
                    is_detection = score <= threshold_;
                }
                
                if (is_detection) {
                    int x1 = j - templ_width_ / 2;
                    int y1 = i - templ_height_ / 2;
                    candidates.emplace_back(x1, y1, templ_width_, templ_height_, score);
                }
            }
        }
        
        // Apply NMS
        auto keep = non_max_suppression(candidates, nms_threshold_);
        std::vector<Detection> result;
        for (int idx : keep) {
            result.push_back(candidates[idx]);
        }
        
        return result;
    }
    
private:
    std::vector<float> template_;
    int templ_width_, templ_height_;
    float threshold_;
    TemplateMethod method_;
    float nms_threshold_;
};

} // namespace detection
} // namespace neurova

#endif // NEUROVA_DETECTION_TEMPLATE_MATCHING_HPP
