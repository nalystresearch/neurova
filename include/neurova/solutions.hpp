// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file solutions.hpp
 * @brief Main header for all high-level solutions
 * 
 * Includes all pre-built computer vision solutions:
 * - Face Detection
 * - Face Mesh (468 landmarks)
 * - Pose Estimation (33 body landmarks)
 * - Hand Tracking (21 landmarks)
 * - Selfie Segmentation
 * - Holistic (combined tracking)
 * - Objectron (3D object detection)
 */

#pragma once

// Solution modules
#include "solutions/face_detection.hpp"
#include "solutions/face_mesh.hpp"
#include "solutions/pose.hpp"
#include "solutions/hands.hpp"
#include "solutions/selfie_segmentation.hpp"
#include "solutions/holistic.hpp"
#include "solutions/objectron.hpp"

namespace neurova {
namespace solutions {

/**
 * @brief Get version of solutions module
 */
inline const char* version() {
    return "1.0.0";
}

/**
 * @brief Solution types
 */
enum class SolutionType {
    FaceDetection,
    FaceMesh,
    Pose,
    Hands,
    SelfieSegmentation,
    Holistic,
    Objectron
};

/**
 * @brief Get solution name
 */
inline const char* solution_name(SolutionType type) {
    switch (type) {
        case SolutionType::FaceDetection: return "Face Detection";
        case SolutionType::FaceMesh: return "Face Mesh";
        case SolutionType::Pose: return "Pose";
        case SolutionType::Hands: return "Hands";
        case SolutionType::SelfieSegmentation: return "Selfie Segmentation";
        case SolutionType::Holistic: return "Holistic";
        case SolutionType::Objectron: return "Objectron";
        default: return "Unknown";
    }
}

/**
 * @brief Get solution description
 */
inline const char* solution_description(SolutionType type) {
    switch (type) {
        case SolutionType::FaceDetection: 
            return "Real-time face detection with key facial landmarks";
        case SolutionType::FaceMesh: 
            return "Dense face mesh with 468 3D landmarks";
        case SolutionType::Pose: 
            return "Full body pose estimation with 33 landmarks";
        case SolutionType::Hands: 
            return "Hand tracking with 21 landmarks per hand";
        case SolutionType::SelfieSegmentation: 
            return "Real-time person segmentation for background effects";
        case SolutionType::Holistic: 
            return "Combined face, pose, and hand tracking";
        case SolutionType::Objectron: 
            return "3D object detection with bounding boxes";
        default: 
            return "Unknown solution";
    }
}

} // namespace solutions
} // namespace neurova
