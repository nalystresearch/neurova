// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file neurova.hpp
 * @brief Main Neurova library header
 * 
 * This is the main header file for the Neurova Computer Vision library.
 * Include this file to get access to all Neurova functionality.
 * 
 * @example
 * @code
 * #include <neurova/neurova.hpp>
 * 
 * int main() {
 *     // Create an image
 *     neurova::Image img(640, 480, 3);
 *     
 *     // Apply transformations
 *     auto resized = neurova::transform::resize(img, 320, 240);
 *     auto gray = neurova::transform::rgb_to_gray(resized);
 *     
 *     // Run face detection
 *     neurova::solutions::FaceDetector detector;
 *     auto faces = detector.detect(img);
 *     
 *     return 0;
 * }
 * @endcode
 */

#pragma once

// Version information
#define NEUROVA_VERSION_MAJOR 1
#define NEUROVA_VERSION_MINOR 0
#define NEUROVA_VERSION_PATCH 0
#define NEUROVA_VERSION_STRING "1.0.0"

// Core modules
#include "core/image.hpp"
#include "core/types.hpp"

// Image transformations
#include "transform.hpp"

// Utility functions
#include "utils.hpp"

// Image processing
#include "morphology.hpp"
#include "segmentation.hpp"
#include "photo.hpp"

// Camera and calibration
#include "calibration.hpp"

// Data augmentation
#include "augmentation.hpp"

// Neural networks
#include "nn.hpp"

// Object detection
#include "object_detection.hpp"

// High-level solutions
#include "solutions.hpp"

namespace neurova {

/**
 * @brief Get Neurova version string
 */
inline const char* version() {
    return NEUROVA_VERSION_STRING;
}

/**
 * @brief Get Neurova version components
 */
inline void version(int& major, int& minor, int& patch) {
    major = NEUROVA_VERSION_MAJOR;
    minor = NEUROVA_VERSION_MINOR;
    patch = NEUROVA_VERSION_PATCH;
}

/**
 * @brief Check if Neurova is built with GPU support
 */
inline bool has_gpu_support() {
#ifdef NEUROVA_WITH_CUDA
    return true;
#elif defined(NEUROVA_WITH_METAL)
    return true;
#elif defined(NEUROVA_WITH_OPENCL)
    return true;
#else
    return false;
#endif
}

/**
 * @brief Get GPU backend name
 */
inline const char* gpu_backend() {
#ifdef NEUROVA_WITH_CUDA
    return "CUDA";
#elif defined(NEUROVA_WITH_METAL)
    return "Metal";
#elif defined(NEUROVA_WITH_OPENCL)
    return "OpenCL";
#else
    return "None";
#endif
}

/**
 * @brief Initialize Neurova library
 */
inline void init() {
    // Placeholder for future initialization
}

/**
 * @brief Cleanup Neurova library resources
 */
inline void shutdown() {
    // Placeholder for future cleanup
}

/**
 * @brief Library information
 */
struct LibraryInfo {
    const char* version = NEUROVA_VERSION_STRING;
    const char* build_type;
    const char* gpu_backend;
    bool has_gpu;
    
    LibraryInfo() {
#ifdef NDEBUG
        build_type = "Release";
#else
        build_type = "Debug";
#endif
        gpu_backend = neurova::gpu_backend();
        has_gpu = neurova::has_gpu_support();
    }
};

/**
 * @brief Get library information
 */
inline LibraryInfo get_library_info() {
    return LibraryInfo();
}

/**
 * @brief Print library information
 */
inline void print_info() {
    auto info = get_library_info();
    // Would print to console in real implementation
}

} // namespace neurova

/**
 * @mainpage Neurova Computer Vision Library
 * 
 * @section intro Introduction
 * 
 * Neurova is a modern C++17 header-only computer vision library designed for
 * high-performance image processing and machine learning inference.
 * 
 * @section features Features
 * 
 * - **Core Image Processing**: Image class with support for multiple color formats
 * - **Transformations**: Resize, crop, rotate, flip, color space conversions
 * - **Neural Networks**: Layers, activations, optimizers for ML inference
 * - **Object Detection**: Anchor generation, NMS, detection heads
 * - **Solutions**: Pre-built face detection, pose estimation, hand tracking
 * - **Augmentation**: Data augmentation for training pipelines
 * - **Calibration**: Camera calibration and pose estimation
 * 
 * @section installation Installation
 * 
 * Neurova is header-only, simply include the headers:
 * @code
 * #include <neurova/neurova.hpp>
 * @endcode
 * 
 * @section modules Modules
 * 
 * - @ref neurova::core - Core image and type definitions
 * - @ref neurova::transform - Image transformations
 * - @ref neurova::nn - Neural network components
 * - @ref neurova::object_detection - Object detection utilities
 * - @ref neurova::solutions - High-level CV solutions
 * - @ref neurova::augmentation - Data augmentation
 * - @ref neurova::calibration - Camera calibration
 * - @ref neurova::morphology - Morphological operations
 * - @ref neurova::segmentation - Image segmentation
 * - @ref neurova::photo - Computational photography
 * 
 * @section license License
 * 
 * Copyright (c) 2026 @squid consultancy group (scg)
 * licensed under the apache license 2.0.
 */
