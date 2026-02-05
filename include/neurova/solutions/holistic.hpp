// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file solutions/holistic.hpp
 * @brief Holistic (face + pose + hands) tracking solution
 * 
 * Neurova implementation of combined multi-modal body tracking.
 */

#pragma once

#include "face_detection.hpp"
#include "face_mesh.hpp"
#include "pose.hpp"
#include "hands.hpp"
#include "../core/image.hpp"
#include <vector>
#include <optional>

namespace neurova {
namespace solutions {

/**
 * @brief Holistic tracking result
 */
struct HolisticResult {
    std::optional<Pose> pose;
    std::optional<FaceMesh> face;
    std::optional<Hand> left_hand;
    std::optional<Hand> right_hand;
    
    // Pose world landmarks (3D in meters)
    std::optional<std::array<PoseLandmark, 33>> pose_world_landmarks;
    
    bool has_pose() const { return pose.has_value(); }
    bool has_face() const { return face.has_value(); }
    bool has_left_hand() const { return left_hand.has_value(); }
    bool has_right_hand() const { return right_hand.has_value(); }
};

/**
 * @brief Holistic tracker configuration
 */
struct HolisticConfig {
    bool enable_face_mesh = true;
    bool enable_segmentation = false;
    bool refine_face_landmarks = true;
    
    PoseDetectorConfig::ModelComplexity model_complexity = PoseDetectorConfig::ModelComplexity::Full;
    float min_detection_confidence = 0.5f;
    float min_tracking_confidence = 0.5f;
};

/**
 * @brief Holistic tracker combining pose, face, and hands
 */
class HolisticTracker {
public:
    explicit HolisticTracker(const HolisticConfig& config = {})
        : config_(config),
          pose_detector_(create_pose_config()),
          face_mesh_detector_(create_face_mesh_config()),
          hand_detector_(create_hand_config()) {}
    
    /**
     * @brief Process image and detect all body parts
     */
    HolisticResult process(const Image& image) {
        HolisticResult result;
        
        // Detect pose first (provides ROIs for face and hands)
        auto poses = pose_detector_.process(image);
        if (!poses.empty()) {
            result.pose = poses[0];
            
            // Extract face ROI from pose
            if (config_.enable_face_mesh) {
                auto face_roi = get_face_roi(poses[0]);
                if (face_roi.width() > 0 && face_roi.height() > 0) {
                    // Detect face mesh
                    auto face_meshes = face_mesh_detector_.process(image);
                    if (!face_meshes.empty()) {
                        result.face = face_meshes[0];
                    }
                }
            }
            
            // Extract hand ROIs from pose
            auto left_hand_roi = get_left_hand_roi(poses[0]);
            auto right_hand_roi = get_right_hand_roi(poses[0]);
            
            // Detect hands
            auto hands = hand_detector_.process(image);
            for (const auto& hand : hands) {
                if (hand.handedness == Handedness::Left && !result.left_hand.has_value()) {
                    result.left_hand = hand;
                } else if (hand.handedness == Handedness::Right && !result.right_hand.has_value()) {
                    result.right_hand = hand;
                }
            }
        }
        
        return result;
    }
    
private:
    HolisticConfig config_;
    PoseDetector pose_detector_;
    FaceMeshDetector face_mesh_detector_;
    HandDetector hand_detector_;
    
    PoseDetectorConfig create_pose_config() {
        PoseDetectorConfig cfg;
        cfg.model_complexity = config_.model_complexity;
        cfg.min_detection_confidence = config_.min_detection_confidence;
        cfg.min_tracking_confidence = config_.min_tracking_confidence;
        return cfg;
    }
    
    FaceMeshConfig create_face_mesh_config() {
        FaceMeshConfig cfg;
        cfg.refine_landmarks = config_.refine_face_landmarks;
        cfg.min_detection_confidence = config_.min_detection_confidence;
        return cfg;
    }
    
    HandDetectorConfig create_hand_config() {
        HandDetectorConfig cfg;
        cfg.max_num_hands = 2;
        cfg.min_detection_confidence = config_.min_detection_confidence;
        return cfg;
    }
    
    object_detection::BBox get_face_roi(const Pose& pose) {
        // Estimate face region from nose and ears
        const auto& nose = pose[PoseLandmarkType::NOSE];
        const auto& left_ear = pose[PoseLandmarkType::LEFT_EAR];
        const auto& right_ear = pose[PoseLandmarkType::RIGHT_EAR];
        
        float cx = nose.x;
        float cy = nose.y;
        float w = std::abs(right_ear.x - left_ear.x) * 2;
        float h = w * 1.2f;
        
        return object_detection::BBox(cx - w/2, cy - h/2, cx + w/2, cy + h/2);
    }
    
    object_detection::BBox get_left_hand_roi(const Pose& pose) {
        const auto& wrist = pose[PoseLandmarkType::LEFT_WRIST];
        const auto& index = pose[PoseLandmarkType::LEFT_INDEX];
        
        float size = std::sqrt(std::pow(index.x - wrist.x, 2) + std::pow(index.y - wrist.y, 2)) * 2;
        return object_detection::BBox(wrist.x - size/2, wrist.y - size/2, 
                                       wrist.x + size/2, wrist.y + size/2);
    }
    
    object_detection::BBox get_right_hand_roi(const Pose& pose) {
        const auto& wrist = pose[PoseLandmarkType::RIGHT_WRIST];
        const auto& index = pose[PoseLandmarkType::RIGHT_INDEX];
        
        float size = std::sqrt(std::pow(index.x - wrist.x, 2) + std::pow(index.y - wrist.y, 2)) * 2;
        return object_detection::BBox(wrist.x - size/2, wrist.y - size/2, 
                                       wrist.x + size/2, wrist.y + size/2);
    }
};

/**
 * @brief Draw holistic result on image
 */
inline void draw_holistic(Image& image, const HolisticResult& result,
                          bool draw_pose = true,
                          bool draw_face = true,
                          bool draw_hands = true) {
    if (draw_pose && result.has_pose()) {
        draw_pose(image, *result.pose);
    }
    
    if (draw_face && result.has_face()) {
        draw_face_mesh(image, *result.face);
    }
    
    if (draw_hands) {
        if (result.has_left_hand()) {
            draw_hand(image, *result.left_hand);
        }
        if (result.has_right_hand()) {
            draw_hand(image, *result.right_hand);
        }
    }
}

} // namespace solutions
} // namespace neurova
