// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file keypoint.hpp
 * @brief Keypoint and feature types
 */

#ifndef NEUROVA_FEATURES_KEYPOINT_HPP
#define NEUROVA_FEATURES_KEYPOINT_HPP

#include <vector>
#include <cmath>
#include <algorithm>

namespace neurova {
namespace features {

/**
 * @brief Keypoint structure for feature detection
 */
struct KeyPoint {
    float x = 0.0f;           ///< X coordinate
    float y = 0.0f;           ///< Y coordinate
    float size = 0.0f;        ///< Diameter of meaningful neighborhood
    float angle = -1.0f;      ///< Orientation in degrees (0-360, -1 if N/A)
    float response = 0.0f;    ///< Response (strength) of keypoint
    int octave = 0;           ///< Octave (pyramid layer)
    int class_id = -1;        ///< Object class (for matching)

    KeyPoint() = default;
    
    KeyPoint(float x_, float y_, float size_ = 0.0f, float angle_ = -1.0f,
             float response_ = 0.0f, int octave_ = 0, int class_id_ = -1)
        : x(x_), y(y_), size(size_), angle(angle_),
          response(response_), octave(octave_), class_id(class_id_) {}

    /// Get point as (x, y) pair
    std::pair<float, float> pt() const { return {x, y}; }
    
    /// Compute distance to another keypoint
    float distance(const KeyPoint& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }
    
    /// Check if keypoint overlaps with another (based on size)
    bool overlaps(const KeyPoint& other, float threshold = 0.5f) const {
        float dist = distance(other);
        float combined_radius = (size + other.size) * 0.5f * threshold;
        return dist < combined_radius;
    }
};

/**
 * @brief Match between two descriptors
 */
struct DMatch {
    int queryIdx = -1;     ///< Query descriptor index
    int trainIdx = -1;     ///< Train descriptor index
    int imgIdx = -1;       ///< Train image index
    float distance = std::numeric_limits<float>::infinity();

    DMatch() = default;
    
    DMatch(int query, int train, float dist)
        : queryIdx(query), trainIdx(train), imgIdx(0), distance(dist) {}
    
    DMatch(int query, int train, int img, float dist)
        : queryIdx(query), trainIdx(train), imgIdx(img), distance(dist) {}

    bool operator<(const DMatch& other) const {
        return distance < other.distance;
    }
};

// Score types
constexpr int HARRIS_SCORE = 0;
constexpr int FAST_SCORE = 1;

// AKAZE descriptor types
constexpr int AKAZE_DESCRIPTOR_KAZE_UPRIGHT = 2;
constexpr int AKAZE_DESCRIPTOR_KAZE = 3;
constexpr int AKAZE_DESCRIPTOR_MLDB_UPRIGHT = 4;
constexpr int AKAZE_DESCRIPTOR_MLDB = 5;

// Norm types for matching
constexpr int NORM_INF = 1;
constexpr int NORM_L1 = 2;
constexpr int NORM_L2 = 4;
constexpr int NORM_L2SQR = 5;
constexpr int NORM_HAMMING = 6;
constexpr int NORM_HAMMING2 = 7;

// Draw flags
constexpr int DRAW_MATCHES_FLAGS_DEFAULT = 0;
constexpr int DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG = 1;
constexpr int DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS = 2;
constexpr int DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS = 4;

// FAST detector types
constexpr int FAST_FEATURE_DETECTOR_TYPE_5_8 = 0;
constexpr int FAST_FEATURE_DETECTOR_TYPE_7_12 = 1;
constexpr int FAST_FEATURE_DETECTOR_TYPE_9_16 = 2;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Non-maximum suppression for keypoints
 * @param keypoints Input keypoints
 * @param radius Suppression radius
 * @return Filtered keypoints
 */
inline std::vector<KeyPoint> keypointNMS(
    const std::vector<KeyPoint>& keypoints,
    float radius = 10.0f
) {
    if (keypoints.empty()) return {};
    
    // Sort by response (strongest first)
    std::vector<KeyPoint> sorted = keypoints;
    std::sort(sorted.begin(), sorted.end(),
              [](const KeyPoint& a, const KeyPoint& b) {
                  return a.response > b.response;
              });
    
    std::vector<KeyPoint> result;
    std::vector<bool> suppressed(sorted.size(), false);
    
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (suppressed[i]) continue;
        
        result.push_back(sorted[i]);
        
        // Suppress nearby keypoints
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (suppressed[j]) continue;
            
            float dx = sorted[i].x - sorted[j].x;
            float dy = sorted[i].y - sorted[j].y;
            float dist = std::sqrt(dx * dx + dy * dy);
            
            if (dist < radius) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

/**
 * @brief Filter keypoints by response threshold
 * @param keypoints Input keypoints
 * @param threshold Minimum response value
 * @return Filtered keypoints
 */
inline std::vector<KeyPoint> filterByResponse(
    const std::vector<KeyPoint>& keypoints,
    float threshold
) {
    std::vector<KeyPoint> result;
    result.reserve(keypoints.size());
    
    for (const auto& kp : keypoints) {
        if (kp.response >= threshold) {
            result.push_back(kp);
        }
    }
    
    return result;
}

/**
 * @brief Keep top N keypoints by response
 * @param keypoints Input keypoints
 * @param n Number of keypoints to keep
 * @return Top N keypoints
 */
inline std::vector<KeyPoint> keepTopN(
    std::vector<KeyPoint> keypoints,
    size_t n
) {
    if (keypoints.size() <= n) return keypoints;
    
    // Partial sort to get top N
    std::partial_sort(
        keypoints.begin(),
        keypoints.begin() + n,
        keypoints.end(),
        [](const KeyPoint& a, const KeyPoint& b) {
            return a.response > b.response;
        }
    );
    
    keypoints.resize(n);
    return keypoints;
}

/**
 * @brief Filter keypoints within image bounds
 * @param keypoints Input keypoints
 * @param width Image width
 * @param height Image height
 * @param border Border margin
 * @return Filtered keypoints
 */
inline std::vector<KeyPoint> filterByBounds(
    const std::vector<KeyPoint>& keypoints,
    int width,
    int height,
    int border = 0
) {
    std::vector<KeyPoint> result;
    result.reserve(keypoints.size());
    
    for (const auto& kp : keypoints) {
        if (kp.x >= border && kp.x < width - border &&
            kp.y >= border && kp.y < height - border) {
            result.push_back(kp);
        }
    }
    
    return result;
}

/**
 * @brief Compute average keypoint response
 * @param keypoints Input keypoints
 * @return Average response value
 */
inline float averageResponse(const std::vector<KeyPoint>& keypoints) {
    if (keypoints.empty()) return 0.0f;
    
    float sum = 0.0f;
    for (const auto& kp : keypoints) {
        sum += kp.response;
    }
    
    return sum / static_cast<float>(keypoints.size());
}

/**
 * @brief Convert keypoints to point vector
 * @param keypoints Input keypoints
 * @return Vector of (x, y) coordinates
 */
inline std::vector<std::pair<float, float>> keypointsToPoints(
    const std::vector<KeyPoint>& keypoints
) {
    std::vector<std::pair<float, float>> points;
    points.reserve(keypoints.size());
    
    for (const auto& kp : keypoints) {
        points.push_back({kp.x, kp.y});
    }
    
    return points;
}

} // namespace features
} // namespace neurova

#endif // NEUROVA_FEATURES_KEYPOINT_HPP
