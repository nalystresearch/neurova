// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file solutions/face_mesh.hpp
 * @brief Face mesh (468 landmarks) solution
 * 
 * Neurova implementation of dense face landmark detection.
 */

#pragma once

#include "../core/image.hpp"
#include "face_detection.hpp"
#include <vector>
#include <array>
#include <cmath>

namespace neurova {
namespace solutions {

/**
 * @brief 3D face landmark
 */
struct FaceLandmark3D {
    float x = 0, y = 0, z = 0;
    float visibility = 1.0f;
    
    FaceLandmark3D() = default;
    FaceLandmark3D(float x_, float y_, float z_, float vis = 1.0f) 
        : x(x_), y(y_), z(z_), visibility(vis) {}
    
    FaceLandmark to_2d() const { return FaceLandmark(x, y, visibility); }
};

/**
 * @brief Face mesh result with 468 landmarks
 */
struct FaceMesh {
    std::vector<FaceLandmark3D> landmarks;  // 468 landmarks
    object_detection::BBox face_bbox;
    float confidence = 0;
    
    // Standard landmark indices (standard compatible)
    static constexpr int LEFT_EYE_CENTER = 468;  // Actually pupil
    static constexpr int RIGHT_EYE_CENTER = 473;
    static constexpr int NOSE_TIP = 1;
    static constexpr int UPPER_LIP = 13;
    static constexpr int LOWER_LIP = 14;
    static constexpr int LEFT_EYE_OUTER = 33;
    static constexpr int LEFT_EYE_INNER = 133;
    static constexpr int RIGHT_EYE_INNER = 362;
    static constexpr int RIGHT_EYE_OUTER = 263;
    static constexpr int LEFT_EYEBROW_OUTER = 70;
    static constexpr int RIGHT_EYEBROW_OUTER = 300;
    
    // Face contour indices (17 points)
    static const std::vector<int>& face_oval_indices() {
        static const std::vector<int> indices = {
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        };
        return indices;
    }
    
    // Eye contour indices
    static const std::vector<int>& left_eye_indices() {
        static const std::vector<int> indices = {
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        };
        return indices;
    }
    
    static const std::vector<int>& right_eye_indices() {
        static const std::vector<int> indices = {
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        };
        return indices;
    }
    
    // Lip contour indices
    static const std::vector<int>& lips_indices() {
        static const std::vector<int> indices = {
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
            14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409
        };
        return indices;
    }
    
    // Get landmark at index with bounds checking
    FaceLandmark3D at(int index) const {
        if (index >= 0 && index < static_cast<int>(landmarks.size())) {
            return landmarks[index];
        }
        return FaceLandmark3D();
    }
    
    // Calculate face orientation (yaw, pitch, roll)
    std::array<float, 3> estimate_pose() const {
        if (landmarks.size() < 468) return {0, 0, 0};
        
        // Use key landmarks to estimate pose
        auto nose = landmarks[NOSE_TIP];
        auto left_eye = landmarks[LEFT_EYE_OUTER];
        auto right_eye = landmarks[RIGHT_EYE_OUTER];
        
        // Calculate yaw from eye positions
        float eye_dx = right_eye.x - left_eye.x;
        float eye_dy = right_eye.y - left_eye.y;
        float roll = std::atan2(eye_dy, eye_dx) * 180.0f / 3.14159265f;
        
        // Calculate yaw from nose offset
        float eye_center_x = (left_eye.x + right_eye.x) / 2;
        float nose_offset = nose.x - eye_center_x;
        float yaw = std::atan2(nose_offset, eye_dx) * 180.0f / 3.14159265f;
        
        // Calculate pitch from vertical positions
        float eye_center_y = (left_eye.y + right_eye.y) / 2;
        float pitch = std::atan2(nose.y - eye_center_y, eye_dx) * 180.0f / 3.14159265f;
        
        return {yaw, pitch, roll};
    }
};

/**
 * @brief Face mesh detector configuration
 */
struct FaceMeshConfig {
    bool refine_landmarks = true;  // Refine eye and lip landmarks
    int max_num_faces = 1;
    float min_detection_confidence = 0.5f;
    float min_tracking_confidence = 0.5f;
};

/**
 * @brief Face mesh detector
 */
class FaceMeshDetector {
public:
    explicit FaceMeshDetector(const FaceMeshConfig& config = {}) : config_(config) {}
    
    /**
     * @brief Detect face mesh landmarks
     */
    std::vector<FaceMesh> process(const Image& image) {
        std::vector<FaceMesh> results;
        
        // First detect faces
        FaceDetectorConfig face_config;
        face_config.confidence_threshold = config_.min_detection_confidence;
        face_config.max_faces = config_.max_num_faces;
        FaceDetector face_detector(face_config);
        
        auto faces = face_detector.detect_haar(image);  // Use Haar for demo
        
        // For each face, run landmark detection
        for (const auto& face : faces) {
            FaceMesh mesh;
            mesh.face_bbox = face.bbox;
            mesh.confidence = face.confidence;
            
            // Extract face region
            int x1 = static_cast<int>(std::max(0.0f, face.bbox.x1));
            int y1 = static_cast<int>(std::max(0.0f, face.bbox.y1));
            int x2 = static_cast<int>(std::min(static_cast<float>(image.width()), face.bbox.x2));
            int y2 = static_cast<int>(std::min(static_cast<float>(image.height()), face.bbox.y2));
            
            float face_w = face.bbox.width();
            float face_h = face.bbox.height();
            
            // Generate placeholder 468 landmarks
            // Real implementation would run neural network inference
            mesh.landmarks.resize(468);
            
            // Initialize with basic face shape
            init_face_mesh_landmarks(mesh, face.bbox);
            
            results.push_back(mesh);
        }
        
        return results;
    }
    
private:
    FaceMeshConfig config_;
    
    void init_face_mesh_landmarks(FaceMesh& mesh, const object_detection::BBox& bbox) {
        // Initialize landmarks in a basic face shape pattern
        float cx = bbox.center_x();
        float cy = bbox.center_y();
        float w = bbox.width();
        float h = bbox.height();
        
        // Place landmarks in approximate positions
        for (int i = 0; i < 468; ++i) {
            // Spiral pattern for demonstration
            float angle = i * 0.1f;
            float radius = (i % 50) * w / 100.0f;
            
            float x = cx + radius * std::cos(angle);
            float y = cy + radius * std::sin(angle);
            float z = (i % 10) * 0.01f;  // Slight depth variation
            
            mesh.landmarks[i] = FaceLandmark3D(x, y, z, 1.0f);
        }
        
        // Override key landmarks with more accurate positions
        // Eyes
        float eye_y = cy - h * 0.1f;
        mesh.landmarks[FaceMesh::LEFT_EYE_OUTER] = FaceLandmark3D(cx - w * 0.25f, eye_y, 0);
        mesh.landmarks[FaceMesh::LEFT_EYE_INNER] = FaceLandmark3D(cx - w * 0.1f, eye_y, 0);
        mesh.landmarks[FaceMesh::RIGHT_EYE_INNER] = FaceLandmark3D(cx + w * 0.1f, eye_y, 0);
        mesh.landmarks[FaceMesh::RIGHT_EYE_OUTER] = FaceLandmark3D(cx + w * 0.25f, eye_y, 0);
        
        // Nose
        mesh.landmarks[FaceMesh::NOSE_TIP] = FaceLandmark3D(cx, cy + h * 0.1f, 0.05f);
        
        // Lips
        mesh.landmarks[FaceMesh::UPPER_LIP] = FaceLandmark3D(cx, cy + h * 0.25f, 0);
        mesh.landmarks[FaceMesh::LOWER_LIP] = FaceLandmark3D(cx, cy + h * 0.3f, 0);
    }
};

/**
 * @brief Draw face mesh on image
 */
inline void draw_face_mesh(Image& image, const FaceMesh& mesh,
                           bool draw_contours = true,
                           bool draw_tesselation = false) {
    // Draw all landmarks as points
    for (const auto& lm : mesh.landmarks) {
        int x = static_cast<int>(lm.x);
        int y = static_cast<int>(lm.y);
        
        if (x >= 0 && x < image.width() && y >= 0 && y < image.height()) {
            if (image.channels() >= 3) {
                image.at(x, y, 0) = 255;
                image.at(x, y, 1) = 255;
                image.at(x, y, 2) = 255;
            }
        }
    }
    
    if (draw_contours) {
        // Draw face oval
        auto draw_contour = [&](const std::vector<int>& indices, float r, float g, float b) {
            for (size_t i = 0; i < indices.size(); ++i) {
                int idx1 = indices[i];
                int idx2 = indices[(i + 1) % indices.size()];
                
                if (idx1 < static_cast<int>(mesh.landmarks.size()) &&
                    idx2 < static_cast<int>(mesh.landmarks.size())) {
                    
                    int x1 = static_cast<int>(mesh.landmarks[idx1].x);
                    int y1 = static_cast<int>(mesh.landmarks[idx1].y);
                    int x2 = static_cast<int>(mesh.landmarks[idx2].x);
                    int y2 = static_cast<int>(mesh.landmarks[idx2].y);
                    
                    // Draw line (Bresenham)
                    int dx = std::abs(x2 - x1);
                    int dy = std::abs(y2 - y1);
                    int sx = x1 < x2 ? 1 : -1;
                    int sy = y1 < y2 ? 1 : -1;
                    int err = dx - dy;
                    
                    while (true) {
                        if (x1 >= 0 && x1 < image.width() && y1 >= 0 && y1 < image.height()) {
                            if (image.channels() >= 3) {
                                image.at(x1, y1, 0) = r;
                                image.at(x1, y1, 1) = g;
                                image.at(x1, y1, 2) = b;
                            }
                        }
                        if (x1 == x2 && y1 == y2) break;
                        int e2 = 2 * err;
                        if (e2 > -dy) { err -= dy; x1 += sx; }
                        if (e2 < dx) { err += dx; y1 += sy; }
                    }
                }
            }
        };
        
        // Draw different parts with different colors
        draw_contour(FaceMesh::face_oval_indices(), 100, 200, 100);
        draw_contour(FaceMesh::left_eye_indices(), 200, 200, 100);
        draw_contour(FaceMesh::right_eye_indices(), 200, 200, 100);
        draw_contour(FaceMesh::lips_indices(), 200, 100, 100);
    }
}

} // namespace solutions
} // namespace neurova
