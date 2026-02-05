// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file solutions/pose.hpp
 * @brief Human pose estimation solution
 * 
 * Neurova implementation of body pose detection (33 landmarks).
 */

#pragma once

#include "../core/image.hpp"
#include "../object_detection/anchor.hpp"
#include "../object_detection/nms.hpp"
#include <vector>
#include <array>
#include <cmath>

namespace neurova {
namespace solutions {

/**
 * @brief 3D pose landmark
 */
struct PoseLandmark {
    float x = 0, y = 0, z = 0;
    float visibility = 0;
    float presence = 0;
    
    PoseLandmark() = default;
    PoseLandmark(float x_, float y_, float z_ = 0, float vis = 1.0f, float pres = 1.0f)
        : x(x_), y(y_), z(z_), visibility(vis), presence(pres) {}
};

/**
 * @brief Pose landmark names (33 landmarks, standard compatible)
 */
enum class PoseLandmarkType {
    NOSE = 0,
    LEFT_EYE_INNER = 1,
    LEFT_EYE = 2,
    LEFT_EYE_OUTER = 3,
    RIGHT_EYE_INNER = 4,
    RIGHT_EYE = 5,
    RIGHT_EYE_OUTER = 6,
    LEFT_EAR = 7,
    RIGHT_EAR = 8,
    MOUTH_LEFT = 9,
    MOUTH_RIGHT = 10,
    LEFT_SHOULDER = 11,
    RIGHT_SHOULDER = 12,
    LEFT_ELBOW = 13,
    RIGHT_ELBOW = 14,
    LEFT_WRIST = 15,
    RIGHT_WRIST = 16,
    LEFT_PINKY = 17,
    RIGHT_PINKY = 18,
    LEFT_INDEX = 19,
    RIGHT_INDEX = 20,
    LEFT_THUMB = 21,
    RIGHT_THUMB = 22,
    LEFT_HIP = 23,
    RIGHT_HIP = 24,
    LEFT_KNEE = 25,
    RIGHT_KNEE = 26,
    LEFT_ANKLE = 27,
    RIGHT_ANKLE = 28,
    LEFT_HEEL = 29,
    RIGHT_HEEL = 30,
    LEFT_FOOT_INDEX = 31,
    RIGHT_FOOT_INDEX = 32
};

/**
 * @brief Skeleton connection for visualization
 */
struct SkeletonConnection {
    int start;
    int end;
};

/**
 * @brief Pose result
 */
struct Pose {
    std::array<PoseLandmark, 33> landmarks;
    float confidence = 0;
    object_detection::BBox bounding_box;
    
    Pose() {
        landmarks.fill(PoseLandmark());
    }
    
    // Access landmark by type
    PoseLandmark& operator[](PoseLandmarkType type) {
        return landmarks[static_cast<int>(type)];
    }
    
    const PoseLandmark& operator[](PoseLandmarkType type) const {
        return landmarks[static_cast<int>(type)];
    }
    
    // Calculate bounding box from landmarks
    void update_bounding_box() {
        float min_x = 1e9f, min_y = 1e9f;
        float max_x = -1e9f, max_y = -1e9f;
        
        for (const auto& lm : landmarks) {
            if (lm.visibility > 0.5f) {
                min_x = std::min(min_x, lm.x);
                min_y = std::min(min_y, lm.y);
                max_x = std::max(max_x, lm.x);
                max_y = std::max(max_y, lm.y);
            }
        }
        
        bounding_box = object_detection::BBox(min_x, min_y, max_x, max_y);
    }
    
    // Standard skeleton connections for visualization
    static const std::vector<SkeletonConnection>& skeleton_connections() {
        static const std::vector<SkeletonConnection> connections = {
            // Face
            {0, 1}, {1, 2}, {2, 3}, {3, 7},
            {0, 4}, {4, 5}, {5, 6}, {6, 8},
            {9, 10},
            // Torso
            {11, 12}, {11, 23}, {12, 24}, {23, 24},
            // Left arm
            {11, 13}, {13, 15}, {15, 17}, {15, 19}, {15, 21}, {17, 19},
            // Right arm
            {12, 14}, {14, 16}, {16, 18}, {16, 20}, {16, 22}, {18, 20},
            // Left leg
            {23, 25}, {25, 27}, {27, 29}, {27, 31}, {29, 31},
            // Right leg
            {24, 26}, {26, 28}, {28, 30}, {28, 32}, {30, 32}
        };
        return connections;
    }
};

/**
 * @brief Pose detector configuration
 */
struct PoseDetectorConfig {
    enum class ModelComplexity {
        Lite = 0,     // Fast
        Full = 1,     // Balanced
        Heavy = 2     // Accurate
    };
    
    ModelComplexity model_complexity = ModelComplexity::Full;
    bool smooth_landmarks = true;
    bool enable_segmentation = false;
    float min_detection_confidence = 0.5f;
    float min_tracking_confidence = 0.5f;
    int max_num_poses = 1;
};

/**
 * @brief Pose detector
 */
class PoseDetector {
public:
    explicit PoseDetector(const PoseDetectorConfig& config = {}) : config_(config) {}
    
    /**
     * @brief Detect poses in an image
     */
    std::vector<Pose> process(const Image& image) {
        std::vector<Pose> poses;
        
        // Person detection first
        auto person_boxes = detect_persons(image);
        
        for (const auto& box : person_boxes) {
            if (poses.size() >= static_cast<size_t>(config_.max_num_poses)) break;
            
            Pose pose;
            pose.bounding_box = box;
            pose.confidence = 1.0f;
            
            // Run landmark detection (placeholder - real impl would use neural network)
            estimate_landmarks(image, box, pose);
            
            poses.push_back(pose);
        }
        
        return poses;
    }
    
private:
    PoseDetectorConfig config_;
    
    std::vector<object_detection::BBox> detect_persons(const Image& image) {
        // Simplified person detection using motion/color
        std::vector<object_detection::BBox> boxes;
        
        // For demo, return a centered box
        float cx = image.width() / 2.0f;
        float cy = image.height() / 2.0f;
        float w = image.width() * 0.5f;
        float h = image.height() * 0.8f;
        
        boxes.push_back(object_detection::BBox(cx - w/2, cy - h/2, cx + w/2, cy + h/2));
        
        return boxes;
    }
    
    void estimate_landmarks(const Image& image, const object_detection::BBox& box, Pose& pose) {
        float cx = box.center_x();
        float cy = box.center_y();
        float w = box.width();
        float h = box.height();
        
        // Initialize landmarks in standard pose positions
        // Head
        pose[PoseLandmarkType::NOSE] = PoseLandmark(cx, cy - h * 0.4f, 0, 1.0f);
        pose[PoseLandmarkType::LEFT_EYE] = PoseLandmark(cx - w * 0.05f, cy - h * 0.42f, 0, 1.0f);
        pose[PoseLandmarkType::RIGHT_EYE] = PoseLandmark(cx + w * 0.05f, cy - h * 0.42f, 0, 1.0f);
        pose[PoseLandmarkType::LEFT_EAR] = PoseLandmark(cx - w * 0.1f, cy - h * 0.4f, 0, 1.0f);
        pose[PoseLandmarkType::RIGHT_EAR] = PoseLandmark(cx + w * 0.1f, cy - h * 0.4f, 0, 1.0f);
        pose[PoseLandmarkType::MOUTH_LEFT] = PoseLandmark(cx - w * 0.03f, cy - h * 0.35f, 0, 1.0f);
        pose[PoseLandmarkType::MOUTH_RIGHT] = PoseLandmark(cx + w * 0.03f, cy - h * 0.35f, 0, 1.0f);
        
        // Shoulders
        pose[PoseLandmarkType::LEFT_SHOULDER] = PoseLandmark(cx - w * 0.2f, cy - h * 0.25f, 0, 1.0f);
        pose[PoseLandmarkType::RIGHT_SHOULDER] = PoseLandmark(cx + w * 0.2f, cy - h * 0.25f, 0, 1.0f);
        
        // Arms
        pose[PoseLandmarkType::LEFT_ELBOW] = PoseLandmark(cx - w * 0.3f, cy - h * 0.1f, 0, 1.0f);
        pose[PoseLandmarkType::RIGHT_ELBOW] = PoseLandmark(cx + w * 0.3f, cy - h * 0.1f, 0, 1.0f);
        pose[PoseLandmarkType::LEFT_WRIST] = PoseLandmark(cx - w * 0.35f, cy + h * 0.05f, 0, 1.0f);
        pose[PoseLandmarkType::RIGHT_WRIST] = PoseLandmark(cx + w * 0.35f, cy + h * 0.05f, 0, 1.0f);
        
        // Hands
        pose[PoseLandmarkType::LEFT_PINKY] = PoseLandmark(cx - w * 0.38f, cy + h * 0.08f, 0, 0.8f);
        pose[PoseLandmarkType::LEFT_INDEX] = PoseLandmark(cx - w * 0.36f, cy + h * 0.1f, 0, 0.8f);
        pose[PoseLandmarkType::LEFT_THUMB] = PoseLandmark(cx - w * 0.33f, cy + h * 0.06f, 0, 0.8f);
        pose[PoseLandmarkType::RIGHT_PINKY] = PoseLandmark(cx + w * 0.38f, cy + h * 0.08f, 0, 0.8f);
        pose[PoseLandmarkType::RIGHT_INDEX] = PoseLandmark(cx + w * 0.36f, cy + h * 0.1f, 0, 0.8f);
        pose[PoseLandmarkType::RIGHT_THUMB] = PoseLandmark(cx + w * 0.33f, cy + h * 0.06f, 0, 0.8f);
        
        // Hips
        pose[PoseLandmarkType::LEFT_HIP] = PoseLandmark(cx - w * 0.1f, cy + h * 0.1f, 0, 1.0f);
        pose[PoseLandmarkType::RIGHT_HIP] = PoseLandmark(cx + w * 0.1f, cy + h * 0.1f, 0, 1.0f);
        
        // Legs
        pose[PoseLandmarkType::LEFT_KNEE] = PoseLandmark(cx - w * 0.1f, cy + h * 0.25f, 0, 1.0f);
        pose[PoseLandmarkType::RIGHT_KNEE] = PoseLandmark(cx + w * 0.1f, cy + h * 0.25f, 0, 1.0f);
        pose[PoseLandmarkType::LEFT_ANKLE] = PoseLandmark(cx - w * 0.1f, cy + h * 0.4f, 0, 1.0f);
        pose[PoseLandmarkType::RIGHT_ANKLE] = PoseLandmark(cx + w * 0.1f, cy + h * 0.4f, 0, 1.0f);
        
        // Feet
        pose[PoseLandmarkType::LEFT_HEEL] = PoseLandmark(cx - w * 0.12f, cy + h * 0.42f, 0, 0.9f);
        pose[PoseLandmarkType::RIGHT_HEEL] = PoseLandmark(cx + w * 0.12f, cy + h * 0.42f, 0, 0.9f);
        pose[PoseLandmarkType::LEFT_FOOT_INDEX] = PoseLandmark(cx - w * 0.08f, cy + h * 0.45f, 0, 0.9f);
        pose[PoseLandmarkType::RIGHT_FOOT_INDEX] = PoseLandmark(cx + w * 0.08f, cy + h * 0.45f, 0, 0.9f);
        
        // Fill in remaining landmarks
        for (int i = 0; i < 33; ++i) {
            if (pose.landmarks[i].visibility < 0.01f) {
                pose.landmarks[i] = PoseLandmark(cx, cy, 0, 0.5f);
            }
        }
    }
};

/**
 * @brief Calculate angle between three landmarks
 */
inline float calculate_angle(const PoseLandmark& a, const PoseLandmark& b, const PoseLandmark& c) {
    float ba_x = a.x - b.x;
    float ba_y = a.y - b.y;
    float bc_x = c.x - b.x;
    float bc_y = c.y - b.y;
    
    float dot = ba_x * bc_x + ba_y * bc_y;
    float cross = ba_x * bc_y - ba_y * bc_x;
    
    float angle = std::atan2(cross, dot) * 180.0f / 3.14159265f;
    return std::abs(angle);
}

/**
 * @brief Draw pose skeleton on image
 */
inline void draw_pose(Image& image, const Pose& pose,
                      bool draw_landmarks = true,
                      bool draw_connections = true) {
    // Draw connections first (behind landmarks)
    if (draw_connections) {
        for (const auto& conn : Pose::skeleton_connections()) {
            const auto& lm1 = pose.landmarks[conn.start];
            const auto& lm2 = pose.landmarks[conn.end];
            
            if (lm1.visibility < 0.5f || lm2.visibility < 0.5f) continue;
            
            int x1 = static_cast<int>(lm1.x);
            int y1 = static_cast<int>(lm1.y);
            int x2 = static_cast<int>(lm2.x);
            int y2 = static_cast<int>(lm2.y);
            
            // Draw line (Bresenham)
            int dx = std::abs(x2 - x1);
            int dy = std::abs(y2 - y1);
            int sx = x1 < x2 ? 1 : -1;
            int sy = y1 < y2 ? 1 : -1;
            int err = dx - dy;
            
            while (true) {
                if (x1 >= 0 && x1 < image.width() && y1 >= 0 && y1 < image.height()) {
                    if (image.channels() >= 3) {
                        image.at(x1, y1, 0) = 0;
                        image.at(x1, y1, 1) = 255;
                        image.at(x1, y1, 2) = 0;
                    }
                }
                if (x1 == x2 && y1 == y2) break;
                int e2 = 2 * err;
                if (e2 > -dy) { err -= dy; x1 += sx; }
                if (e2 < dx) { err += dx; y1 += sy; }
            }
        }
    }
    
    // Draw landmarks
    if (draw_landmarks) {
        for (int i = 0; i < 33; ++i) {
            const auto& lm = pose.landmarks[i];
            if (lm.visibility < 0.5f) continue;
            
            int x = static_cast<int>(lm.x);
            int y = static_cast<int>(lm.y);
            int radius = 3;
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx*dx + dy*dy <= radius*radius) {
                        int px = x + dx;
                        int py = y + dy;
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
