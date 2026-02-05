// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file descriptors.hpp
 * @brief Feature descriptors (ORB, SIFT-like, AKAZE-like)
 */

#ifndef NEUROVA_FEATURES_DESCRIPTORS_HPP
#define NEUROVA_FEATURES_DESCRIPTORS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <memory>

#include "keypoint.hpp"
#include "corners.hpp"

namespace neurova {
namespace features {

// ============================================================================
// Feature2D Base Class
// ============================================================================

/**
 * @brief Base class for 2D feature detectors and descriptors
 */
class Feature2D {
public:
    virtual ~Feature2D() = default;
    
    /**
     * @brief Detect keypoints
     */
    virtual std::vector<KeyPoint> detect(
        const float* image, int width, int height,
        const uint8_t* mask = nullptr
    ) = 0;
    
    /**
     * @brief Compute descriptors for keypoints
     */
    virtual std::vector<std::vector<uint8_t>> compute(
        const float* image, int width, int height,
        std::vector<KeyPoint>& keypoints
    ) = 0;
    
    /**
     * @brief Detect keypoints and compute descriptors
     */
    virtual std::pair<std::vector<KeyPoint>, std::vector<std::vector<uint8_t>>>
    detectAndCompute(
        const float* image, int width, int height,
        const uint8_t* mask = nullptr
    ) {
        auto keypoints = detect(image, width, height, mask);
        auto descriptors = compute(image, width, height, keypoints);
        return {keypoints, descriptors};
    }
};

// ============================================================================
// FAST Corner Detector
// ============================================================================

/**
 * @brief FAST feature detector
 */
class FastFeatureDetector {
public:
    FastFeatureDetector(
        int threshold = 10,
        bool nonmaxSuppression = true,
        int type = FAST_FEATURE_DETECTOR_TYPE_9_16
    ) : threshold_(threshold),
        nonmaxSuppression_(nonmaxSuppression),
        type_(type) {}

    std::vector<KeyPoint> detect(
        const float* gray, int width, int height,
        const uint8_t* mask = nullptr
    ) {
        std::vector<KeyPoint> keypoints;
        std::vector<float> responses(width * height, 0.0f);
        
        // FAST-16 circle pattern
        static const int circle16[16][2] = {
            {0, 3}, {1, 3}, {2, 2}, {3, 1},
            {3, 0}, {3, -1}, {2, -2}, {1, -3},
            {0, -3}, {-1, -3}, {-2, -2}, {-3, -1},
            {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}
        };
        
        int nContiguous = (type_ == FAST_FEATURE_DETECTOR_TYPE_9_16) ? 9 : 7;
        
        for (int y = 3; y < height - 3; ++y) {
            for (int x = 3; x < width - 3; ++x) {
                if (mask && !mask[y * width + x]) continue;
                
                float center = gray[y * width + x];
                float thresh = static_cast<float>(threshold_);
                
                // Get circle pixels
                float circleVals[16];
                for (int i = 0; i < 16; ++i) {
                    circleVals[i] = gray[(y + circle16[i][1]) * width + (x + circle16[i][0])];
                }
                
                // Check for contiguous brighter/darker
                bool isCorner = false;
                for (int start = 0; start < 16 && !isCorner; ++start) {
                    int brightCount = 0, darkCount = 0;
                    for (int i = 0; i < nContiguous; ++i) {
                        int idx = (start + i) % 16;
                        if (circleVals[idx] > center + thresh) ++brightCount;
                        if (circleVals[idx] < center - thresh) ++darkCount;
                    }
                    if (brightCount == nContiguous || darkCount == nContiguous) {
                        isCorner = true;
                    }
                }
                
                if (isCorner) {
                    float response = 0.0f;
                    for (int i = 0; i < 16; ++i) {
                        float diff = std::abs(circleVals[i] - center);
                        if (diff > thresh) response += diff;
                    }
                    responses[y * width + x] = response;
                    
                    if (!nonmaxSuppression_) {
                        keypoints.emplace_back(static_cast<float>(x), static_cast<float>(y),
                                              7.0f, -1.0f, response);
                    }
                }
            }
        }
        
        // Non-maximum suppression
        if (nonmaxSuppression_) {
            for (int y = 4; y < height - 4; ++y) {
                for (int x = 4; x < width - 4; ++x) {
                    float r = responses[y * width + x];
                    if (r <= 0) continue;
                    
                    bool isMax = true;
                    for (int dy = -1; dy <= 1 && isMax; ++dy) {
                        for (int dx = -1; dx <= 1 && isMax; ++dx) {
                            if (dy == 0 && dx == 0) continue;
                            if (responses[(y + dy) * width + (x + dx)] >= r) {
                                isMax = false;
                            }
                        }
                    }
                    
                    if (isMax) {
                        keypoints.emplace_back(static_cast<float>(x), static_cast<float>(y),
                                              7.0f, -1.0f, r);
                    }
                }
            }
        }
        
        return keypoints;
    }

    int getThreshold() const { return threshold_; }
    void setThreshold(int t) { threshold_ = t; }

private:
    int threshold_;
    bool nonmaxSuppression_;
    int type_;
};

// ============================================================================
// ORB Detector and Descriptor
// ============================================================================

/**
 * @brief ORB (Oriented FAST and Rotated BRIEF) feature detector and descriptor
 */
class ORB : public Feature2D {
public:
    ORB(int nfeatures = 500,
        float scaleFactor = 1.2f,
        int nlevels = 8,
        int edgeThreshold = 31,
        int firstLevel = 0,
        int WTA_K = 2,
        int scoreType = HARRIS_SCORE,
        int patchSize = 31,
        int fastThreshold = 20)
        : nfeatures_(nfeatures),
          scaleFactor_(scaleFactor),
          nlevels_(nlevels),
          edgeThreshold_(edgeThreshold),
          firstLevel_(firstLevel),
          WTA_K_(WTA_K),
          scoreType_(scoreType),
          patchSize_(patchSize),
          fastThreshold_(fastThreshold) {
        generateBriefPattern();
    }

    std::vector<KeyPoint> detect(
        const float* image, int width, int height,
        const uint8_t* mask = nullptr
    ) override {
        FastFeatureDetector fast(fastThreshold_, true);
        auto keypoints = fast.detect(image, width, height, mask);
        
        // Filter by edge threshold
        std::vector<KeyPoint> filtered;
        for (const auto& kp : keypoints) {
            if (kp.x >= edgeThreshold_ && kp.x < width - edgeThreshold_ &&
                kp.y >= edgeThreshold_ && kp.y < height - edgeThreshold_) {
                filtered.push_back(kp);
            }
        }
        
        // Sort by response and keep top N
        std::sort(filtered.begin(), filtered.end(),
                  [](const auto& a, const auto& b) {
                      return a.response > b.response;
                  });
        
        if (static_cast<int>(filtered.size()) > nfeatures_) {
            filtered.resize(nfeatures_);
        }
        
        // Compute orientation
        for (auto& kp : filtered) {
            kp.angle = computeOrientation(image, width, height, 
                                         static_cast<int>(kp.x), 
                                         static_cast<int>(kp.y));
        }
        
        return filtered;
    }

    std::vector<std::vector<uint8_t>> compute(
        const float* image, int width, int height,
        std::vector<KeyPoint>& keypoints
    ) override {
        std::vector<std::vector<uint8_t>> descriptors;
        descriptors.reserve(keypoints.size());
        
        for (const auto& kp : keypoints) {
            auto desc = computeBriefDescriptor(image, width, height,
                                              static_cast<int>(kp.x),
                                              static_cast<int>(kp.y),
                                              kp.angle);
            descriptors.push_back(desc);
        }
        
        return descriptors;
    }

    static std::unique_ptr<ORB> create(
        int nfeatures = 500,
        float scaleFactor = 1.2f,
        int nlevels = 8,
        int edgeThreshold = 31,
        int firstLevel = 0,
        int WTA_K = 2,
        int scoreType = HARRIS_SCORE,
        int patchSize = 31,
        int fastThreshold = 20
    ) {
        return std::make_unique<ORB>(nfeatures, scaleFactor, nlevels,
                                     edgeThreshold, firstLevel, WTA_K,
                                     scoreType, patchSize, fastThreshold);
    }

private:
    int nfeatures_;
    float scaleFactor_;
    int nlevels_;
    int edgeThreshold_;
    int firstLevel_;
    int WTA_K_;
    int scoreType_;
    int patchSize_;
    int fastThreshold_;
    std::vector<std::array<int, 4>> briefPattern_;

    void generateBriefPattern() {
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(-patchSize_ / 2, patchSize_ / 2);
        
        briefPattern_.resize(256);
        for (int i = 0; i < 256; ++i) {
            briefPattern_[i] = {dist(gen), dist(gen), dist(gen), dist(gen)};
        }
    }

    float computeOrientation(const float* image, int w, int h, int x, int y) {
        int radius = patchSize_ / 2;
        float m01 = 0.0f, m10 = 0.0f;
        
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = std::clamp(x + dx, 0, w - 1);
                int ny = std::clamp(y + dy, 0, h - 1);
                float val = image[ny * w + nx];
                m01 += dy * val;
                m10 += dx * val;
            }
        }
        
        return std::atan2(m01, m10) * 180.0f / 3.14159265f;
    }

    std::vector<uint8_t> computeBriefDescriptor(
        const float* image, int w, int h, int x, int y, float angle
    ) {
        std::vector<uint8_t> descriptor(32, 0);
        
        float rad = angle * 3.14159265f / 180.0f;
        float cosA = std::cos(rad);
        float sinA = std::sin(rad);
        
        for (int i = 0; i < 256; ++i) {
            const auto& pat = briefPattern_[i];
            
            // Rotate pattern points
            int x1 = static_cast<int>(pat[0] * cosA - pat[1] * sinA);
            int y1 = static_cast<int>(pat[0] * sinA + pat[1] * cosA);
            int x2 = static_cast<int>(pat[2] * cosA - pat[3] * sinA);
            int y2 = static_cast<int>(pat[2] * sinA + pat[3] * cosA);
            
            // Sample
            int nx1 = std::clamp(x + x1, 0, w - 1);
            int ny1 = std::clamp(y + y1, 0, h - 1);
            int nx2 = std::clamp(x + x2, 0, w - 1);
            int ny2 = std::clamp(y + y2, 0, h - 1);
            
            float v1 = image[ny1 * w + nx1];
            float v2 = image[ny2 * w + nx2];
            
            if (v1 < v2) {
                descriptor[i / 8] |= (1 << (i % 8));
            }
        }
        
        return descriptor;
    }
};

// Factory function
inline std::unique_ptr<ORB> ORB_create(
    int nfeatures = 500,
    float scaleFactor = 1.2f,
    int nlevels = 8,
    int edgeThreshold = 31,
    int firstLevel = 0,
    int WTA_K = 2,
    int scoreType = HARRIS_SCORE,
    int patchSize = 31,
    int fastThreshold = 20
) {
    return ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                       firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
}

// ============================================================================
// SIFT-like Descriptor (Simplified)
// ============================================================================

/**
 * @brief Simplified SIFT-like feature detector
 */
class SIFT : public Feature2D {
public:
    SIFT(int nfeatures = 0,
         int nOctaveLayers = 3,
         double contrastThreshold = 0.04,
         double edgeThreshold = 10,
         double sigma = 1.6)
        : nfeatures_(nfeatures),
          nOctaveLayers_(nOctaveLayers),
          contrastThreshold_(contrastThreshold),
          edgeThreshold_(edgeThreshold),
          sigma_(sigma) {}

    std::vector<KeyPoint> detect(
        const float* image, int width, int height,
        const uint8_t* mask = nullptr
    ) override {
        // Use Harris corners as keypoint locations
        auto response = harrisResponse(image, width, height);
        auto corners = nonMaxSuppression(response, width, height, 
                                        static_cast<float>(contrastThreshold_),
                                        5, nfeatures_ > 0 ? nfeatures_ : 500);
        
        std::vector<KeyPoint> keypoints;
        for (const auto& [x, y] : corners) {
            if (mask && !mask[y * width + x]) continue;
            keypoints.emplace_back(static_cast<float>(x), static_cast<float>(y),
                                  static_cast<float>(sigma_), -1.0f,
                                  response[y * width + x]);
        }
        
        return keypoints;
    }

    std::vector<std::vector<uint8_t>> compute(
        const float* image, int width, int height,
        std::vector<KeyPoint>& keypoints
    ) override {
        // Compute gradient magnitude and orientation
        std::vector<float> gx, gy;
        sobel(image, width, height, gx, gy);
        
        std::vector<std::vector<uint8_t>> descriptors;
        descriptors.reserve(keypoints.size());
        
        for (const auto& kp : keypoints) {
            auto desc = computeSiftDescriptor(gx.data(), gy.data(), width, height,
                                             static_cast<int>(kp.x),
                                             static_cast<int>(kp.y));
            descriptors.push_back(desc);
        }
        
        return descriptors;
    }

    static std::unique_ptr<SIFT> create(
        int nfeatures = 0,
        int nOctaveLayers = 3,
        double contrastThreshold = 0.04,
        double edgeThreshold = 10,
        double sigma = 1.6
    ) {
        return std::make_unique<SIFT>(nfeatures, nOctaveLayers,
                                      contrastThreshold, edgeThreshold, sigma);
    }

private:
    int nfeatures_;
    int nOctaveLayers_;
    double contrastThreshold_;
    double edgeThreshold_;
    double sigma_;

    std::vector<uint8_t> computeSiftDescriptor(
        const float* gx, const float* gy, int w, int h, int x, int y
    ) {
        // 4x4 grid of 8-bin histograms = 128 dimensions
        std::vector<float> histograms(128, 0.0f);
        int patchSize = 16;
        int cellSize = 4;
        int nBins = 8;
        
        for (int cy = 0; cy < 4; ++cy) {
            for (int cx = 0; cx < 4; ++cx) {
                for (int dy = 0; dy < cellSize; ++dy) {
                    for (int dx = 0; dx < cellSize; ++dx) {
                        int px = x - patchSize/2 + cx * cellSize + dx;
                        int py = y - patchSize/2 + cy * cellSize + dy;
                        
                        if (px < 0 || px >= w || py < 0 || py >= h) continue;
                        
                        float mag = std::sqrt(gx[py * w + px] * gx[py * w + px] +
                                             gy[py * w + px] * gy[py * w + px]);
                        float angle = std::atan2(gy[py * w + px], gx[py * w + px]);
                        angle = angle * 180.0f / 3.14159265f + 180.0f;
                        
                        int bin = static_cast<int>(angle / 45.0f) % nBins;
                        int histIdx = (cy * 4 + cx) * nBins + bin;
                        histograms[histIdx] += mag;
                    }
                }
            }
        }
        
        // Normalize and convert to uint8
        float norm = 0.0f;
        for (float h : histograms) norm += h * h;
        norm = std::sqrt(norm) + 1e-7f;
        
        std::vector<uint8_t> descriptor(128);
        for (int i = 0; i < 128; ++i) {
            float val = std::min(histograms[i] / norm, 0.2f);
            descriptor[i] = static_cast<uint8_t>(std::min(255.0f, val * 512.0f));
        }
        
        return descriptor;
    }
};

inline std::unique_ptr<SIFT> SIFT_create(
    int nfeatures = 0,
    int nOctaveLayers = 3,
    double contrastThreshold = 0.04,
    double edgeThreshold = 10,
    double sigma = 1.6
) {
    return SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
}

// ============================================================================
// AKAZE-like Descriptor (Simplified)
// ============================================================================

/**
 * @brief Simplified AKAZE-like feature detector
 */
class AKAZE : public Feature2D {
public:
    AKAZE(int descriptorType = AKAZE_DESCRIPTOR_MLDB,
          int descriptorSize = 0,
          int descriptorChannels = 3,
          float threshold = 0.001f,
          int nOctaves = 4,
          int nOctaveLayers = 4)
        : descriptorType_(descriptorType),
          descriptorSize_(descriptorSize),
          descriptorChannels_(descriptorChannels),
          threshold_(threshold),
          nOctaves_(nOctaves),
          nOctaveLayers_(nOctaveLayers) {}

    std::vector<KeyPoint> detect(
        const float* image, int width, int height,
        const uint8_t* mask = nullptr
    ) override {
        auto response = shiTomasiResponse(image, width, height);
        auto corners = nonMaxSuppression(response, width, height,
                                        threshold_, 5, 1000);
        
        std::vector<KeyPoint> keypoints;
        for (const auto& [x, y] : corners) {
            if (mask && !mask[y * width + x]) continue;
            keypoints.emplace_back(static_cast<float>(x), static_cast<float>(y),
                                  5.0f, -1.0f, response[y * width + x]);
        }
        
        return keypoints;
    }

    std::vector<std::vector<uint8_t>> compute(
        const float* image, int width, int height,
        std::vector<KeyPoint>& keypoints
    ) override {
        std::vector<std::vector<uint8_t>> descriptors;
        descriptors.reserve(keypoints.size());
        
        for (const auto& kp : keypoints) {
            // MLDB-like binary descriptor
            auto desc = computeMLDBDescriptor(image, width, height,
                                             static_cast<int>(kp.x),
                                             static_cast<int>(kp.y));
            descriptors.push_back(desc);
        }
        
        return descriptors;
    }

    static std::unique_ptr<AKAZE> create(
        int descriptorType = AKAZE_DESCRIPTOR_MLDB,
        int descriptorSize = 0,
        int descriptorChannels = 3,
        float threshold = 0.001f,
        int nOctaves = 4,
        int nOctaveLayers = 4
    ) {
        return std::make_unique<AKAZE>(descriptorType, descriptorSize,
                                       descriptorChannels, threshold,
                                       nOctaves, nOctaveLayers);
    }

private:
    int descriptorType_;
    int descriptorSize_;
    int descriptorChannels_;
    float threshold_;
    int nOctaves_;
    int nOctaveLayers_;

    std::vector<uint8_t> computeMLDBDescriptor(
        const float* image, int w, int h, int x, int y
    ) {
        // 61-byte (486-bit) descriptor
        std::vector<uint8_t> descriptor(61, 0);
        int bitIdx = 0;
        
        // Multi-scale comparisons
        for (int scale = 1; scale <= 3; ++scale) {
            int step = scale * 2;
            for (int dy = -4; dy <= 4; dy += step) {
                for (int dx = -4; dx <= 4; dx += step) {
                    if (dx == 0 && dy == 0) continue;
                    
                    int x1 = std::clamp(x + dx, 0, w - 1);
                    int y1 = std::clamp(y + dy, 0, h - 1);
                    int x2 = std::clamp(x - dx, 0, w - 1);
                    int y2 = std::clamp(y - dy, 0, h - 1);
                    
                    if (image[y1 * w + x1] < image[y2 * w + x2]) {
                        descriptor[bitIdx / 8] |= (1 << (bitIdx % 8));
                    }
                    
                    ++bitIdx;
                    if (bitIdx >= 486) break;
                }
                if (bitIdx >= 486) break;
            }
            if (bitIdx >= 486) break;
        }
        
        return descriptor;
    }
};

inline std::unique_ptr<AKAZE> AKAZE_create(
    int descriptorType = AKAZE_DESCRIPTOR_MLDB,
    int descriptorSize = 0,
    int descriptorChannels = 3,
    float threshold = 0.001f,
    int nOctaves = 4,
    int nOctaveLayers = 4
) {
    return AKAZE::create(descriptorType, descriptorSize, descriptorChannels,
                         threshold, nOctaves, nOctaveLayers);
}

} // namespace features
} // namespace neurova

#endif // NEUROVA_FEATURES_DESCRIPTORS_HPP
