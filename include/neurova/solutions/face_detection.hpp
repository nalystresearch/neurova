// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file solutions/face_detection.hpp
 * @brief Face detection solution
 * 
 * Neurova implementation of face detection pipeline.
 */

#pragma once

#include "../core/image.hpp"
#include "../object_detection/anchor.hpp"
#include "../object_detection/nms.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace neurova {
namespace solutions {

/**
 * @brief Face detection landmark (keypoint)
 */
struct FaceLandmark {
    float x = 0, y = 0;
    float visibility = 1.0f;
    
    FaceLandmark() = default;
    FaceLandmark(float x_, float y_, float vis = 1.0f) : x(x_), y(y_), visibility(vis) {}
};

/**
 * @brief Face detection result
 */
struct Face {
    object_detection::BBox bbox;
    float confidence = 0;
    std::vector<FaceLandmark> landmarks;  // Typically 5 or 68 landmarks
    
    Face() = default;
    Face(const object_detection::BBox& b, float conf) : bbox(b), confidence(conf) {}
    
    // Get 5 key points (eyes, nose, mouth corners)
    FaceLandmark left_eye() const { return landmarks.size() > 0 ? landmarks[0] : FaceLandmark(); }
    FaceLandmark right_eye() const { return landmarks.size() > 1 ? landmarks[1] : FaceLandmark(); }
    FaceLandmark nose() const { return landmarks.size() > 2 ? landmarks[2] : FaceLandmark(); }
    FaceLandmark mouth_left() const { return landmarks.size() > 3 ? landmarks[3] : FaceLandmark(); }
    FaceLandmark mouth_right() const { return landmarks.size() > 4 ? landmarks[4] : FaceLandmark(); }
};

/**
 * @brief Face detector configuration
 */
struct FaceDetectorConfig {
    int input_size = 640;
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.4f;
    int max_faces = 100;
    bool detect_landmarks = true;
    
    // Model selection
    enum class Model {
        BlazeFace,      // Fast, mobile-friendly
        RetinaFace,     // Accurate, landmark detection
        SCRFD           // SOTA accuracy
    };
    Model model = Model::BlazeFace;
};

/**
 * @brief Face detector
 */
class FaceDetector {
public:
    explicit FaceDetector(const FaceDetectorConfig& config = {}) : config_(config) {
        init_anchors();
    }
    
    /**
     * @brief Detect faces in an image
     */
    std::vector<Face> detect(const Image& image) {
        // Preprocessing
        float scale = static_cast<float>(config_.input_size) / std::max(image.width(), image.height());
        int new_w = static_cast<int>(image.width() * scale);
        int new_h = static_cast<int>(image.height() * scale);
        
        // Pad to square
        int pad_w = config_.input_size - new_w;
        int pad_h = config_.input_size - new_h;
        
        // Run inference (placeholder - actual inference would use TFLite/ONNX)
        auto raw_detections = run_inference(image);
        
        // Apply NMS
        auto keep = object_detection::nms(raw_detections, config_.nms_threshold);
        
        // Convert to Face objects and rescale
        std::vector<Face> faces;
        faces.reserve(keep.size());
        
        for (int idx : keep) {
            if (faces.size() >= static_cast<size_t>(config_.max_faces)) break;
            
            Face face(raw_detections[idx].bbox, raw_detections[idx].confidence);
            
            // Rescale to original image
            float inv_scale = 1.0f / scale;
            face.bbox.x1 = face.bbox.x1 * inv_scale;
            face.bbox.y1 = face.bbox.y1 * inv_scale;
            face.bbox.x2 = face.bbox.x2 * inv_scale;
            face.bbox.y2 = face.bbox.y2 * inv_scale;
            
            // Clip to image bounds
            face.bbox = face.bbox.clip(image.width(), image.height());
            
            faces.push_back(face);
        }
        
        return faces;
    }
    
    /**
     * @brief Detect faces using Haar cascade (fallback/lightweight)
     */
    std::vector<Face> detect_haar(const Image& image, float scale_factor = 1.1f,
                                   int min_neighbors = 3, int min_size = 30) {
        std::vector<Face> faces;
        
        // Convert to grayscale
        Image gray(image.width(), image.height(), 1);
        if (image.channels() >= 3) {
            for (int y = 0; y < image.height(); ++y) {
                for (int x = 0; x < image.width(); ++x) {
                    gray.at(x, y, 0) = 0.299f * image.at(x, y, 0) +
                                       0.587f * image.at(x, y, 1) +
                                       0.114f * image.at(x, y, 2);
                }
            }
        } else {
            gray = image;
        }
        
        // Compute integral image
        std::vector<std::vector<int>> integral(gray.height() + 1, 
                                               std::vector<int>(gray.width() + 1, 0));
        for (int y = 0; y < gray.height(); ++y) {
            int row_sum = 0;
            for (int x = 0; x < gray.width(); ++x) {
                row_sum += static_cast<int>(gray.at(x, y, 0));
                integral[y + 1][x + 1] = integral[y][x + 1] + row_sum;
            }
        }
        
        // Multi-scale sliding window (simplified Haar cascade)
        std::vector<object_detection::Detection> candidates;
        
        for (float scale = 1.0f; scale * min_size < std::min(gray.width(), gray.height()); 
             scale *= scale_factor) {
            
            int win_size = static_cast<int>(min_size * scale);
            int step = std::max(1, win_size / 10);
            
            for (int y = 0; y + win_size < gray.height(); y += step) {
                for (int x = 0; x + win_size < gray.width(); x += step) {
                    // Compute features (simplified: just variance-based)
                    float response = compute_haar_response(integral, x, y, win_size);
                    
                    if (response > config_.confidence_threshold) {
                        object_detection::Detection det;
                        det.bbox = object_detection::BBox(
                            static_cast<float>(x), static_cast<float>(y),
                            static_cast<float>(x + win_size), static_cast<float>(y + win_size));
                        det.confidence = response;
                        det.class_id = 0;
                        candidates.push_back(det);
                    }
                }
            }
        }
        
        // Group overlapping detections
        auto keep = object_detection::nms(candidates, config_.nms_threshold);
        
        for (int idx : keep) {
            Face face(candidates[idx].bbox, candidates[idx].confidence);
            faces.push_back(face);
        }
        
        return faces;
    }
    
private:
    FaceDetectorConfig config_;
    std::vector<object_detection::BBox> anchors_;
    
    void init_anchors() {
        // Generate anchors for face detection
        std::vector<int> strides = {8, 16, 32};
        std::vector<std::vector<float>> anchor_sizes = {
            {16, 32}, {64, 128}, {256, 512}
        };
        
        for (size_t i = 0; i < strides.size(); ++i) {
            int stride = strides[i];
            int grid_size = config_.input_size / stride;
            
            for (int y = 0; y < grid_size; ++y) {
                for (int x = 0; x < grid_size; ++x) {
                    float cx = (x + 0.5f) * stride;
                    float cy = (y + 0.5f) * stride;
                    
                    for (float size : anchor_sizes[i]) {
                        anchors_.push_back(object_detection::BBox::from_cxcywh(cx, cy, size, size));
                    }
                }
            }
        }
    }
    
    std::vector<object_detection::Detection> run_inference(const Image& image) {
        // Placeholder - actual implementation would run neural network inference
        std::vector<object_detection::Detection> detections;
        return detections;
    }
    
    float compute_haar_response(const std::vector<std::vector<int>>& integral,
                                 int x, int y, int size) {
        // Simplified Haar-like feature response
        int half = size / 2;
        int quarter = size / 4;
        
        // Eye region (darker)
        float eye_region = rect_sum(integral, x + quarter, y + quarter, half, quarter);
        
        // Cheek region (lighter)
        float cheek_region = rect_sum(integral, x + quarter, y + half, half, quarter);
        
        // Normalize by area
        float eye_mean = eye_region / (half * quarter);
        float cheek_mean = cheek_region / (half * quarter);
        
        // Response: eyes darker than cheeks
        float response = (cheek_mean - eye_mean) / 255.0f;
        return std::max(0.0f, std::min(1.0f, response * 2.0f));
    }
    
    float rect_sum(const std::vector<std::vector<int>>& integral,
                   int x, int y, int w, int h) {
        return static_cast<float>(
            integral[y + h][x + w] - integral[y][x + w] -
            integral[y + h][x] + integral[y][x]);
    }
};

/**
 * @brief Draw faces on image
 */
inline void draw_faces(Image& image, const std::vector<Face>& faces,
                       float r = 0, float g = 255, float b = 0, int thickness = 2) {
    for (const auto& face : faces) {
        int x1 = static_cast<int>(face.bbox.x1);
        int y1 = static_cast<int>(face.bbox.y1);
        int x2 = static_cast<int>(face.bbox.x2);
        int y2 = static_cast<int>(face.bbox.y2);
        
        // Draw rectangle
        for (int t = 0; t < thickness; ++t) {
            // Top and bottom
            for (int x = x1; x <= x2; ++x) {
                if (y1 + t >= 0 && y1 + t < image.height() && x >= 0 && x < image.width()) {
                    if (image.channels() >= 3) {
                        image.at(x, y1 + t, 0) = r;
                        image.at(x, y1 + t, 1) = g;
                        image.at(x, y1 + t, 2) = b;
                    }
                }
                if (y2 - t >= 0 && y2 - t < image.height() && x >= 0 && x < image.width()) {
                    if (image.channels() >= 3) {
                        image.at(x, y2 - t, 0) = r;
                        image.at(x, y2 - t, 1) = g;
                        image.at(x, y2 - t, 2) = b;
                    }
                }
            }
            // Left and right
            for (int y = y1; y <= y2; ++y) {
                if (x1 + t >= 0 && x1 + t < image.width() && y >= 0 && y < image.height()) {
                    if (image.channels() >= 3) {
                        image.at(x1 + t, y, 0) = r;
                        image.at(x1 + t, y, 1) = g;
                        image.at(x1 + t, y, 2) = b;
                    }
                }
                if (x2 - t >= 0 && x2 - t < image.width() && y >= 0 && y < image.height()) {
                    if (image.channels() >= 3) {
                        image.at(x2 - t, y, 0) = r;
                        image.at(x2 - t, y, 1) = g;
                        image.at(x2 - t, y, 2) = b;
                    }
                }
            }
        }
        
        // Draw landmarks
        for (const auto& lm : face.landmarks) {
            int lx = static_cast<int>(lm.x);
            int ly = static_cast<int>(lm.y);
            
            // Draw small circle
            for (int dy = -2; dy <= 2; ++dy) {
                for (int dx = -2; dx <= 2; ++dx) {
                    if (dx*dx + dy*dy <= 4) {
                        int px = lx + dx;
                        int py = ly + dy;
                        if (px >= 0 && px < image.width() && py >= 0 && py < image.height()) {
                            if (image.channels() >= 3) {
                                image.at(px, py, 0) = 255;
                                image.at(px, py, 1) = 0;
                                image.at(px, py, 2) = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace solutions
} // namespace neurova
