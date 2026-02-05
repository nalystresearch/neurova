// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file features.hpp
 * @brief Feature detection and matching module
 */

#ifndef NEUROVA_FEATURES_HPP
#define NEUROVA_FEATURES_HPP

#include "features/keypoint.hpp"
#include "features/corners.hpp"
#include "features/descriptors.hpp"
#include "features/matching.hpp"

namespace neurova {
namespace features {

// Additional feature detectors

/**
 * @brief Good Features to Track (Shi-Tomasi corners)
 */
inline std::vector<KeyPoint> goodFeaturesToTrack(
    const float* gray, int width, int height,
    int maxCorners = 100,
    float qualityLevel = 0.01f,
    float minDistance = 10.0f,
    const uint8_t* mask = nullptr,
    int blockSize = 3,
    bool useHarrisDetector = false,
    float k = 0.04f
) {
    std::vector<float> response;
    if (useHarrisDetector) {
        response = harrisResponse(gray, width, height, blockSize, k);
    } else {
        response = shiTomasiResponse(gray, width, height, blockSize);
    }
    
    auto corners = nonMaxSuppression(response, width, height,
                                     qualityLevel, static_cast<int>(minDistance),
                                     maxCorners);
    
    std::vector<KeyPoint> keypoints;
    for (const auto& [x, y] : corners) {
        if (mask && !mask[y * width + x]) continue;
        keypoints.emplace_back(static_cast<float>(x), static_cast<float>(y),
                              1.0f, -1.0f, response[y * width + x]);
    }
    
    return keypoints;
}

/**
 * @brief BRISK feature detector (simplified)
 */
class BRISK : public Feature2D {
public:
    BRISK(int thresh = 30, int octaves = 3, float patternScale = 1.0f)
        : thresh_(thresh), octaves_(octaves), patternScale_(patternScale) {}

    std::vector<KeyPoint> detect(
        const float* image, int width, int height,
        const uint8_t* mask = nullptr
    ) override {
        FastFeatureDetector fast(thresh_, true);
        return fast.detect(image, width, height, mask);
    }

    std::vector<std::vector<uint8_t>> compute(
        const float* image, int width, int height,
        std::vector<KeyPoint>& keypoints
    ) override {
        // BRISK uses pattern-based sampling
        std::vector<std::vector<uint8_t>> descriptors;
        descriptors.reserve(keypoints.size());
        
        for (const auto& kp : keypoints) {
            auto desc = computeDescriptor(image, width, height,
                                         static_cast<int>(kp.x),
                                         static_cast<int>(kp.y));
            descriptors.push_back(desc);
        }
        
        return descriptors;
    }

    static std::unique_ptr<BRISK> create(int thresh = 30, int octaves = 3, 
                                         float patternScale = 1.0f) {
        return std::make_unique<BRISK>(thresh, octaves, patternScale);
    }

private:
    int thresh_;
    int octaves_;
    float patternScale_;

    std::vector<uint8_t> computeDescriptor(
        const float* image, int w, int h, int x, int y
    ) {
        // 64-byte descriptor
        std::vector<uint8_t> descriptor(64, 0);
        
        // Concentric circles pattern
        static const float radii[] = {0, 2.9f, 4.9f, 7.4f, 10.8f};
        static const int points[] = {1, 10, 14, 15, 20};
        
        int bitIdx = 0;
        for (int r = 0; r < 5; ++r) {
            float radius = radii[r] * patternScale_;
            int nPoints = points[r];
            
            for (int p1 = 0; p1 < nPoints; ++p1) {
                float angle1 = 2.0f * 3.14159265f * p1 / nPoints;
                int px1 = static_cast<int>(x + radius * std::cos(angle1));
                int py1 = static_cast<int>(y + radius * std::sin(angle1));
                
                for (int p2 = p1 + 1; p2 < nPoints; ++p2) {
                    if (bitIdx >= 512) break;
                    
                    float angle2 = 2.0f * 3.14159265f * p2 / nPoints;
                    int px2 = static_cast<int>(x + radius * std::cos(angle2));
                    int py2 = static_cast<int>(y + radius * std::sin(angle2));
                    
                    px1 = std::clamp(px1, 0, w - 1);
                    py1 = std::clamp(py1, 0, h - 1);
                    px2 = std::clamp(px2, 0, w - 1);
                    py2 = std::clamp(py2, 0, h - 1);
                    
                    if (image[py1 * w + px1] < image[py2 * w + px2]) {
                        descriptor[bitIdx / 8] |= (1 << (bitIdx % 8));
                    }
                    ++bitIdx;
                }
            }
        }
        
        return descriptor;
    }
};

/**
 * @brief Simple Blob Detector
 */
class SimpleBlobDetector {
public:
    struct Params {
        float minThreshold = 50.0f;
        float maxThreshold = 220.0f;
        float thresholdStep = 10.0f;
        float minArea = 25.0f;
        float maxArea = 5000.0f;
        float minCircularity = 0.8f;
        float minConvexity = 0.95f;
        float minInertiaRatio = 0.1f;
        bool filterByArea = true;
        bool filterByCircularity = false;
        bool filterByConvexity = false;
        bool filterByInertia = false;
    };

    SimpleBlobDetector(const Params& params = Params())
        : params_(params) {}

    std::vector<KeyPoint> detect(
        const float* gray, int width, int height,
        const uint8_t* mask = nullptr
    ) {
        std::vector<KeyPoint> keypoints;
        
        // Simple blob detection using thresholding
        for (float thresh = params_.minThreshold; 
             thresh < params_.maxThreshold; 
             thresh += params_.thresholdStep) {
            
            // Find connected components at this threshold
            auto blobs = findBlobs(gray, width, height, thresh);
            
            for (const auto& blob : blobs) {
                if (params_.filterByArea) {
                    if (blob.area < params_.minArea || blob.area > params_.maxArea) {
                        continue;
                    }
                }
                
                if (mask && !mask[static_cast<int>(blob.y) * width + 
                                  static_cast<int>(blob.x)]) {
                    continue;
                }
                
                keypoints.emplace_back(blob.x, blob.y, 
                                      std::sqrt(blob.area / 3.14159265f) * 2,
                                      -1.0f, blob.area);
            }
        }
        
        // Remove duplicates
        std::vector<KeyPoint> unique;
        for (const auto& kp : keypoints) {
            bool isDup = false;
            for (const auto& ukp : unique) {
                float dx = kp.x - ukp.x;
                float dy = kp.y - ukp.y;
                if (std::sqrt(dx * dx + dy * dy) < 10.0f) {
                    isDup = true;
                    break;
                }
            }
            if (!isDup) {
                unique.push_back(kp);
            }
        }
        
        return unique;
    }

    static std::unique_ptr<SimpleBlobDetector> create(const Params& params = Params()) {
        return std::make_unique<SimpleBlobDetector>(params);
    }

private:
    Params params_;

    struct Blob {
        float x, y;
        float area;
    };

    std::vector<Blob> findBlobs(
        const float* gray, int width, int height, float threshold
    ) {
        std::vector<Blob> blobs;
        std::vector<bool> visited(width * height, false);
        
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                if (visited[y * width + x]) continue;
                if (gray[y * width + x] <= threshold) continue;
                
                // Flood fill to find blob
                float sumX = 0, sumY = 0;
                int count = 0;
                
                std::vector<std::pair<int, int>> stack;
                stack.emplace_back(x, y);
                
                while (!stack.empty()) {
                    auto [cx, cy] = stack.back();
                    stack.pop_back();
                    
                    if (cx < 0 || cx >= width || cy < 0 || cy >= height) continue;
                    if (visited[cy * width + cx]) continue;
                    if (gray[cy * width + cx] <= threshold) continue;
                    
                    visited[cy * width + cx] = true;
                    sumX += cx;
                    sumY += cy;
                    ++count;
                    
                    stack.emplace_back(cx + 1, cy);
                    stack.emplace_back(cx - 1, cy);
                    stack.emplace_back(cx, cy + 1);
                    stack.emplace_back(cx, cy - 1);
                }
                
                if (count > 0) {
                    blobs.push_back({sumX / count, sumY / count, 
                                    static_cast<float>(count)});
                }
            }
        }
        
        return blobs;
    }
};

/**
 * @brief GFTT (Good Features To Track) Detector wrapper
 */
class GFTTDetector {
public:
    GFTTDetector(int maxCorners = 1000,
                 double qualityLevel = 0.01,
                 double minDistance = 1,
                 int blockSize = 3,
                 bool useHarrisDetector = false,
                 double k = 0.04)
        : maxCorners_(maxCorners),
          qualityLevel_(qualityLevel),
          minDistance_(minDistance),
          blockSize_(blockSize),
          useHarris_(useHarrisDetector),
          k_(k) {}

    std::vector<KeyPoint> detect(
        const float* gray, int width, int height,
        const uint8_t* mask = nullptr
    ) {
        return goodFeaturesToTrack(gray, width, height,
                                   maxCorners_,
                                   static_cast<float>(qualityLevel_),
                                   static_cast<float>(minDistance_),
                                   mask, blockSize_, useHarris_,
                                   static_cast<float>(k_));
    }

    static std::unique_ptr<GFTTDetector> create(
        int maxCorners = 1000,
        double qualityLevel = 0.01,
        double minDistance = 1,
        int blockSize = 3,
        bool useHarrisDetector = false,
        double k = 0.04
    ) {
        return std::make_unique<GFTTDetector>(maxCorners, qualityLevel,
                                              minDistance, blockSize,
                                              useHarrisDetector, k);
    }

private:
    int maxCorners_;
    double qualityLevel_;
    double minDistance_;
    int blockSize_;
    bool useHarris_;
    double k_;
};

/**
 * @brief MSER (Maximally Stable Extremal Regions) - Simplified
 */
class MSER {
public:
    MSER(int delta = 5,
         int minArea = 60,
         int maxArea = 14400,
         double maxVariation = 0.25,
         double minDiversity = 0.2)
        : delta_(delta),
          minArea_(minArea),
          maxArea_(maxArea),
          maxVariation_(maxVariation),
          minDiversity_(minDiversity) {}

    std::vector<KeyPoint> detect(
        const float* gray, int width, int height,
        const uint8_t* mask = nullptr
    ) {
        // Simplified MSER using extremal region detection
        std::vector<KeyPoint> keypoints;
        
        // Sample thresholds
        for (int thresh = 0; thresh < 256; thresh += delta_) {
            auto regions = findRegions(gray, width, height, 
                                       static_cast<float>(thresh));
            
            for (const auto& [cx, cy, area] : regions) {
                if (area < minArea_ || area > maxArea_) continue;
                if (mask && !mask[static_cast<int>(cy) * width + 
                                  static_cast<int>(cx)]) continue;
                
                keypoints.emplace_back(cx, cy, 
                                      std::sqrt(static_cast<float>(area) / 3.14159265f),
                                      -1.0f, static_cast<float>(area));
            }
        }
        
        return keypoints;
    }

    static std::unique_ptr<MSER> create(
        int delta = 5,
        int minArea = 60,
        int maxArea = 14400,
        double maxVariation = 0.25,
        double minDiversity = 0.2
    ) {
        return std::make_unique<MSER>(delta, minArea, maxArea,
                                      maxVariation, minDiversity);
    }

private:
    int delta_;
    int minArea_;
    int maxArea_;
    double maxVariation_;
    double minDiversity_;

    std::vector<std::tuple<float, float, int>> findRegions(
        const float* gray, int w, int h, float threshold
    ) {
        std::vector<std::tuple<float, float, int>> regions;
        std::vector<bool> visited(w * h, false);
        
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                if (visited[y * w + x]) continue;
                if (gray[y * w + x] >= threshold) continue;
                
                float sumX = 0, sumY = 0;
                int count = 0;
                
                std::vector<std::pair<int, int>> stack;
                stack.emplace_back(x, y);
                
                while (!stack.empty()) {
                    auto [cx, cy] = stack.back();
                    stack.pop_back();
                    
                    if (cx < 0 || cx >= w || cy < 0 || cy >= h) continue;
                    if (visited[cy * w + cx]) continue;
                    if (gray[cy * w + cx] >= threshold) continue;
                    
                    visited[cy * w + cx] = true;
                    sumX += cx;
                    sumY += cy;
                    ++count;
                    
                    if (count > maxArea_) break;
                    
                    stack.emplace_back(cx + 1, cy);
                    stack.emplace_back(cx - 1, cy);
                    stack.emplace_back(cx, cy + 1);
                    stack.emplace_back(cx, cy - 1);
                }
                
                if (count >= minArea_ && count <= maxArea_) {
                    regions.emplace_back(sumX / count, sumY / count, count);
                }
            }
        }
        
        return regions;
    }
};

} // namespace features
} // namespace neurova

#endif // NEUROVA_FEATURES_HPP
