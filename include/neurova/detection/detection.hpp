// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file detection.hpp
 * @brief Main include for Neurova Detection module
 * 
 * This header provides object detection algorithms including:
 * - Template matching (NCC, SSD, SAD, CCORR methods)
 * - Haar Cascade classifier
 * - HOG descriptor for pedestrian detection
 * - Non-maximum suppression utilities
 * 
 * @example Basic Template Matching
 * @code
 * #include <neurova/detection/detection.hpp>
 * 
 * using namespace neurova::detection;
 * 
 * // Match template in image
 * std::vector<float> result;
 * match_template(image, img_w, img_h, templ, t_w, t_h, result, TemplateMethod::NCC);
 * 
 * // Find best match
 * auto it = std::max_element(result.begin(), result.end());
 * int best_idx = std::distance(result.begin(), it);
 * int match_x = best_idx % (img_w - t_w + 1);
 * int match_y = best_idx / (img_w - t_w + 1);
 * @endcode
 * 
 * @example Haar Cascade Face Detection
 * @code
 * #include <neurova/detection/detection.hpp>
 * 
 * using namespace neurova::detection;
 * 
 * HaarCascadeClassifier cascade("haarcascade_frontalface_default.xml");
 * 
 * std::vector<std::tuple<int, int, int, int>> faces;
 * cascade.detectMultiScale(gray_image, width, height, faces,
 *                          1.1f,    // scale factor
 *                          3,       // min neighbors
 *                          30, 30); // min size
 * 
 * for (auto& [x, y, w, h] : faces) {
 *     // Draw rectangle around face
 * }
 * @endcode
 * 
 * @example HOG Pedestrian Detection
 * @code
 * #include <neurova/detection/detection.hpp>
 * 
 * using namespace neurova::detection;
 * 
 * HOGDescriptor hog;
 * hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
 * 
 * std::vector<std::tuple<int, int, int, int>> pedestrians;
 * std::vector<float> weights;
 * hog.detectMultiScale(gray_image, width, height, pedestrians, weights);
 * @endcode
 */

#ifndef NEUROVA_DETECTION_HPP
#define NEUROVA_DETECTION_HPP

// Include all detection headers
#include "template_matching.hpp"
#include "haar_cascade.hpp"
#include "hog.hpp"

namespace neurova {
namespace detection {

/**
 * @brief Version information for the detection module
 */
constexpr const char* DETECTION_VERSION = "1.0.0";

/**
 * @brief Convenience struct for detection results with confidence
 */
struct DetectionResult {
    int x, y, width, height;
    float confidence;
    int class_id;
    
    DetectionResult(int x_ = 0, int y_ = 0, int w_ = 0, int h_ = 0, 
                   float conf_ = 0.0f, int cls_ = 0)
        : x(x_), y(y_), width(w_), height(h_), confidence(conf_), class_id(cls_) {}
    
    bool operator<(const DetectionResult& other) const {
        return confidence > other.confidence; // Sort by confidence descending
    }
};

/**
 * @brief Apply non-maximum suppression to detection results
 * 
 * @param detections Input detections
 * @param iou_threshold IoU threshold for suppression (default 0.5)
 * @return Filtered detections
 */
inline std::vector<DetectionResult> nms(
    std::vector<DetectionResult> detections,
    float iou_threshold = 0.5f
) {
    if (detections.empty()) return {};
    
    // Sort by confidence
    std::sort(detections.begin(), detections.end());
    
    std::vector<bool> suppressed(detections.size(), false);
    std::vector<DetectionResult> result;
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        
        result.push_back(detections[i]);
        const auto& det_i = detections[i];
        
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            
            const auto& det_j = detections[j];
            
            // Compute IoU
            int xi1 = std::max(det_i.x, det_j.x);
            int yi1 = std::max(det_i.y, det_j.y);
            int xi2 = std::min(det_i.x + det_i.width, det_j.x + det_j.width);
            int yi2 = std::min(det_i.y + det_i.height, det_j.y + det_j.height);
            
            float intersection = 0.0f;
            if (xi2 > xi1 && yi2 > yi1) {
                intersection = static_cast<float>((xi2 - xi1) * (yi2 - yi1));
            }
            
            float area_i = static_cast<float>(det_i.width * det_i.height);
            float area_j = static_cast<float>(det_j.width * det_j.height);
            float union_area = area_i + area_j - intersection;
            float iou = intersection / std::max(union_area, 1e-6f);
            
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

/**
 * @brief Compute Intersection over Union (IoU) between two boxes
 */
inline float computeIoU(int x1, int y1, int w1, int h1,
                        int x2, int y2, int w2, int h2) {
    int xi1 = std::max(x1, x2);
    int yi1 = std::max(y1, y2);
    int xi2 = std::min(x1 + w1, x2 + w2);
    int yi2 = std::min(y1 + h1, y2 + h2);
    
    if (xi2 <= xi1 || yi2 <= yi1) return 0.0f;
    
    float intersection = static_cast<float>((xi2 - xi1) * (yi2 - yi1));
    float union_area = static_cast<float>(w1 * h1 + w2 * h2) - intersection;
    
    return intersection / std::max(union_area, 1e-6f);
}

/**
 * @brief Scale detection coordinates
 */
inline DetectionResult scaleDetection(const DetectionResult& det, float scale) {
    return DetectionResult(
        static_cast<int>(det.x * scale),
        static_cast<int>(det.y * scale),
        static_cast<int>(det.width * scale),
        static_cast<int>(det.height * scale),
        det.confidence,
        det.class_id
    );
}

} // namespace detection
} // namespace neurova

#endif // NEUROVA_DETECTION_HPP
