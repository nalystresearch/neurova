// Copyright (c) 2026 @squid consultancy group (scg)
// all rights reserved.
// licensed under the apache license 2.0.

/**
 * neurova.hpp - Main header file
 * 
 * Include this file to get access to all Neurova functionality.
 */

#ifndef NEUROVA_HPP
#define NEUROVA_HPP

// Core components
#include "neurova/core.hpp"

// Image processing
#include "neurova/imgproc.hpp"

// Neural networks
#include "neurova/neural.hpp"

// Architectures
#include "neurova/architectures.hpp"

// Machine learning
#include "neurova/ml.hpp"

// Deep neural networks
#include "neurova/dnn/dnn.hpp"

// Face detection and recognition
#include "neurova/face.hpp"

// Face module components (alternative granular include)
#include "neurova/face/face_detector.hpp"
#include "neurova/face/face_recognizer.hpp"
#include "neurova/face/face_utils.hpp"

// Object detection (template matching, Haar cascade, HOG)
#include "neurova/detection/detection.hpp"

// Data loading and datasets
#include "neurova/data/data.hpp"

// Video processing
#include "neurova/video.hpp"

// Version information
#define NEUROVA_VERSION_MAJOR 1
#define NEUROVA_VERSION_MINOR 0
#define NEUROVA_VERSION_PATCH 0
#define NEUROVA_VERSION "1.0.0"

namespace neurova {

// Version functions
inline const char* version() { return NEUROVA_VERSION; }
inline int version_major() { return NEUROVA_VERSION_MAJOR; }
inline int version_minor() { return NEUROVA_VERSION_MINOR; }
inline int version_patch() { return NEUROVA_VERSION_PATCH; }

// Build info
inline const char* build_info() {
#ifdef __clang__
    return "Clang " __clang_version__;
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#elif defined(_MSC_VER)
    return "MSVC " _MSC_VER;
#else
    return "Unknown compiler";
#endif
}

// SIMD support detection
inline bool has_sse() {
#if defined(__SSE__)
    return true;
#else
    return false;
#endif
}

inline bool has_sse2() {
#if defined(__SSE2__)
    return true;
#else
    return false;
#endif
}

inline bool has_avx() {
#if defined(__AVX__)
    return true;
#else
    return false;
#endif
}

inline bool has_avx2() {
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

inline bool has_neon() {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return true;
#else
    return false;
#endif
}

// Print system info
inline void print_info() {
    printf("Neurova %s\n", version());
    printf("Build: %s\n", build_info());
    printf("SIMD: SSE=%d SSE2=%d AVX=%d AVX2=%d NEON=%d\n",
           has_sse(), has_sse2(), has_avx(), has_avx2(), has_neon());
}

} // namespace neurova

#endif // NEUROVA_HPP
