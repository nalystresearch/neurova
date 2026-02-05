// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file solutions/objectron.hpp
 * @brief 3D object detection and tracking solution
 * 
 * Neurova implementation of 3D bounding box detection for objects.
 */

#pragma once

#include "../core/image.hpp"
#include "../object_detection/anchor.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <string>

namespace neurova {
namespace solutions {

/**
 * @brief 3D keypoint
 */
struct Keypoint3D {
    float x = 0, y = 0, z = 0;
    
    Keypoint3D() = default;
    Keypoint3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

/**
 * @brief 3D bounding box with 9 keypoints (8 corners + center)
 */
struct Box3D {
    std::array<Keypoint3D, 9> keypoints;  // 0=center, 1-8=corners
    float rotation[9] = {1,0,0, 0,1,0, 0,0,1};  // 3x3 rotation matrix
    float translation[3] = {0, 0, 0};
    float scale[3] = {1, 1, 1};  // Width, height, depth
    
    // Box corner order:
    // Front face: 1-4 (starting top-left, going clockwise)
    // Back face: 5-8 (starting top-left, going clockwise)
    
    Keypoint3D center() const { return keypoints[0]; }
    
    // Get 2D projection of box
    std::array<object_detection::Point2f, 9> project_2d() const {
        std::array<object_detection::Point2f, 9> projected;
        for (int i = 0; i < 9; ++i) {
            projected[i] = object_detection::Point2f(keypoints[i].x, keypoints[i].y);
        }
        return projected;
    }
    
    // Get edges for drawing
    static const std::vector<std::pair<int, int>>& edges() {
        static const std::vector<std::pair<int, int>> e = {
            // Front face
            {1, 2}, {2, 3}, {3, 4}, {4, 1},
            // Back face
            {5, 6}, {6, 7}, {7, 8}, {8, 5},
            // Connecting edges
            {1, 5}, {2, 6}, {3, 7}, {4, 8}
        };
        return e;
    }
};

/**
 * @brief Objectron result
 */
struct ObjectronResult {
    Box3D box;
    int category_id = -1;
    std::string category_name;
    float confidence = 0;
    object_detection::BBox bbox_2d;  // 2D bounding box
};

/**
 * @brief Supported object categories
 */
enum class ObjectCategory {
    Camera = 0,
    Chair = 1,
    Cup = 2,
    Shoe = 3
};

/**
 * @brief Objectron configuration
 */
struct ObjectronConfig {
    ObjectCategory category = ObjectCategory::Cup;
    int max_num_objects = 5;
    float min_detection_confidence = 0.5f;
    float min_tracking_confidence = 0.5f;
    bool static_image_mode = false;
};

/**
 * @brief Objectron 3D object detector
 */
class Objectron {
public:
    explicit Objectron(const ObjectronConfig& config = {}) : config_(config) {
        category_names_ = {"camera", "chair", "cup", "shoe"};
    }
    
    /**
     * @brief Detect 3D objects in image
     */
    std::vector<ObjectronResult> process(const Image& image) {
        std::vector<ObjectronResult> results;
        
        // Detect 2D objects first
        auto detections = detect_2d(image);
        
        for (const auto& det : detections) {
            if (results.size() >= static_cast<size_t>(config_.max_num_objects)) break;
            
            ObjectronResult result;
            result.bbox_2d = det.bbox;
            result.confidence = det.confidence;
            result.category_id = static_cast<int>(config_.category);
            result.category_name = category_names_[result.category_id];
            
            // Estimate 3D box
            estimate_3d_box(image, det.bbox, result.box);
            
            results.push_back(result);
        }
        
        return results;
    }
    
private:
    ObjectronConfig config_;
    std::vector<std::string> category_names_;
    
    std::vector<object_detection::Detection> detect_2d(const Image& image) {
        std::vector<object_detection::Detection> detections;
        
        // Placeholder - returns centered detection for demo
        object_detection::Detection det;
        det.bbox = object_detection::BBox(
            image.width() * 0.3f, image.height() * 0.3f,
            image.width() * 0.7f, image.height() * 0.7f);
        det.confidence = 0.9f;
        det.class_id = static_cast<int>(config_.category);
        detections.push_back(det);
        
        return detections;
    }
    
    void estimate_3d_box(const Image& image, const object_detection::BBox& bbox, Box3D& box) {
        float cx = bbox.center_x();
        float cy = bbox.center_y();
        float w = bbox.width();
        float h = bbox.height();
        
        // Estimate depth based on object size (simple perspective)
        float estimated_depth = 1.0f / (w / image.width());
        
        // Set center
        box.keypoints[0] = Keypoint3D(cx, cy, estimated_depth);
        
        // Estimate 3D dimensions based on category
        float obj_w, obj_h, obj_d;
        switch (config_.category) {
            case ObjectCategory::Cup:
                obj_w = 0.08f; obj_h = 0.12f; obj_d = 0.08f;
                break;
            case ObjectCategory::Chair:
                obj_w = 0.5f; obj_h = 1.0f; obj_d = 0.5f;
                break;
            case ObjectCategory::Camera:
                obj_w = 0.15f; obj_h = 0.1f; obj_d = 0.08f;
                break;
            case ObjectCategory::Shoe:
                obj_w = 0.1f; obj_h = 0.1f; obj_d = 0.3f;
                break;
        }
        
        // Set corners (approximate projection)
        float half_w = w / 2;
        float half_h = h / 2;
        float depth_offset = h * 0.3f;  // Perspective depth
        
        // Front face corners (1-4)
        box.keypoints[1] = Keypoint3D(cx - half_w, cy - half_h, estimated_depth);
        box.keypoints[2] = Keypoint3D(cx + half_w, cy - half_h, estimated_depth);
        box.keypoints[3] = Keypoint3D(cx + half_w, cy + half_h, estimated_depth);
        box.keypoints[4] = Keypoint3D(cx - half_w, cy + half_h, estimated_depth);
        
        // Back face corners (5-8) - offset in Z
        float back_scale = 0.8f;  // Perspective scaling
        box.keypoints[5] = Keypoint3D(cx - half_w * back_scale, cy - half_h * back_scale - depth_offset, estimated_depth + 0.1f);
        box.keypoints[6] = Keypoint3D(cx + half_w * back_scale, cy - half_h * back_scale - depth_offset, estimated_depth + 0.1f);
        box.keypoints[7] = Keypoint3D(cx + half_w * back_scale, cy + half_h * back_scale - depth_offset, estimated_depth + 0.1f);
        box.keypoints[8] = Keypoint3D(cx - half_w * back_scale, cy + half_h * back_scale - depth_offset, estimated_depth + 0.1f);
        
        // Set scale
        box.scale[0] = obj_w;
        box.scale[1] = obj_h;
        box.scale[2] = obj_d;
        
        // Set translation
        box.translation[0] = cx;
        box.translation[1] = cy;
        box.translation[2] = estimated_depth;
    }
};

/**
 * @brief Draw 3D bounding box on image
 */
inline void draw_box_3d(Image& image, const Box3D& box,
                        float r = 0, float g = 255, float b = 0) {
    auto draw_line = [&](const Keypoint3D& p1, const Keypoint3D& p2) {
        int x1 = static_cast<int>(p1.x);
        int y1 = static_cast<int>(p1.y);
        int x2 = static_cast<int>(p2.x);
        int y2 = static_cast<int>(p2.y);
        
        // Bresenham
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
    };
    
    // Draw all edges
    for (const auto& edge : Box3D::edges()) {
        draw_line(box.keypoints[edge.first], box.keypoints[edge.second]);
    }
    
    // Draw center point
    int cx = static_cast<int>(box.keypoints[0].x);
    int cy = static_cast<int>(box.keypoints[0].y);
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int px = cx + dx;
            int py = cy + dy;
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

} // namespace solutions
} // namespace neurova
