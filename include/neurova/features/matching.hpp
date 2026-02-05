// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file matching.hpp
 * @brief Feature matching algorithms
 */

#ifndef NEUROVA_FEATURES_MATCHING_HPP
#define NEUROVA_FEATURES_MATCHING_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <memory>

#include "keypoint.hpp"

namespace neurova {
namespace features {

// ============================================================================
// Descriptor Matcher Base Class
// ============================================================================

/**
 * @brief Base class for descriptor matchers
 */
class DescriptorMatcher {
public:
    virtual ~DescriptorMatcher() = default;
    
    /**
     * @brief Find best match for each query descriptor
     */
    virtual std::vector<DMatch> match(
        const std::vector<std::vector<uint8_t>>& queryDescriptors,
        const std::vector<std::vector<uint8_t>>& trainDescriptors
    ) = 0;
    
    /**
     * @brief Find k best matches for each query descriptor
     */
    virtual std::vector<std::vector<DMatch>> knnMatch(
        const std::vector<std::vector<uint8_t>>& queryDescriptors,
        const std::vector<std::vector<uint8_t>>& trainDescriptors,
        int k
    ) = 0;
    
    /**
     * @brief Find matches within a radius
     */
    virtual std::vector<std::vector<DMatch>> radiusMatch(
        const std::vector<std::vector<uint8_t>>& queryDescriptors,
        const std::vector<std::vector<uint8_t>>& trainDescriptors,
        float maxDistance
    ) = 0;
};

// ============================================================================
// Brute Force Matcher
// ============================================================================

/**
 * @brief Brute-Force Descriptor Matcher
 */
class BFMatcher : public DescriptorMatcher {
public:
    BFMatcher(int normType = NORM_L2, bool crossCheck = false)
        : normType_(normType), crossCheck_(crossCheck) {}

    std::vector<DMatch> match(
        const std::vector<std::vector<uint8_t>>& queryDescriptors,
        const std::vector<std::vector<uint8_t>>& trainDescriptors
    ) override {
        if (queryDescriptors.empty() || trainDescriptors.empty()) {
            return {};
        }
        
        std::vector<DMatch> matches;
        
        for (size_t i = 0; i < queryDescriptors.size(); ++i) {
            float bestDist = std::numeric_limits<float>::infinity();
            int bestIdx = -1;
            
            for (size_t j = 0; j < trainDescriptors.size(); ++j) {
                float dist = computeDistance(queryDescriptors[i], trainDescriptors[j]);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = static_cast<int>(j);
                }
            }
            
            if (bestIdx >= 0) {
                matches.emplace_back(static_cast<int>(i), bestIdx, bestDist);
            }
        }
        
        if (crossCheck_) {
            // Filter by cross-check
            std::vector<int> reverseMatches(trainDescriptors.size(), -1);
            
            for (size_t j = 0; j < trainDescriptors.size(); ++j) {
                float bestDist = std::numeric_limits<float>::infinity();
                int bestIdx = -1;
                
                for (size_t i = 0; i < queryDescriptors.size(); ++i) {
                    float dist = computeDistance(queryDescriptors[i], trainDescriptors[j]);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestIdx = static_cast<int>(i);
                    }
                }
                
                reverseMatches[j] = bestIdx;
            }
            
            std::vector<DMatch> crossChecked;
            for (const auto& m : matches) {
                if (m.trainIdx >= 0 && 
                    reverseMatches[m.trainIdx] == m.queryIdx) {
                    crossChecked.push_back(m);
                }
            }
            matches = crossChecked;
        }
        
        return matches;
    }

    std::vector<std::vector<DMatch>> knnMatch(
        const std::vector<std::vector<uint8_t>>& queryDescriptors,
        const std::vector<std::vector<uint8_t>>& trainDescriptors,
        int k
    ) override {
        std::vector<std::vector<DMatch>> allMatches;
        
        for (size_t i = 0; i < queryDescriptors.size(); ++i) {
            std::vector<DMatch> kMatches;
            
            for (size_t j = 0; j < trainDescriptors.size(); ++j) {
                float dist = computeDistance(queryDescriptors[i], trainDescriptors[j]);
                kMatches.emplace_back(static_cast<int>(i), static_cast<int>(j), dist);
            }
            
            // Sort and keep top k
            std::sort(kMatches.begin(), kMatches.end());
            if (static_cast<int>(kMatches.size()) > k) {
                kMatches.resize(k);
            }
            
            allMatches.push_back(kMatches);
        }
        
        return allMatches;
    }

    std::vector<std::vector<DMatch>> radiusMatch(
        const std::vector<std::vector<uint8_t>>& queryDescriptors,
        const std::vector<std::vector<uint8_t>>& trainDescriptors,
        float maxDistance
    ) override {
        std::vector<std::vector<DMatch>> allMatches;
        
        for (size_t i = 0; i < queryDescriptors.size(); ++i) {
            std::vector<DMatch> radiusMatches;
            
            for (size_t j = 0; j < trainDescriptors.size(); ++j) {
                float dist = computeDistance(queryDescriptors[i], trainDescriptors[j]);
                if (dist <= maxDistance) {
                    radiusMatches.emplace_back(static_cast<int>(i), static_cast<int>(j), dist);
                }
            }
            
            std::sort(radiusMatches.begin(), radiusMatches.end());
            allMatches.push_back(radiusMatches);
        }
        
        return allMatches;
    }

    static std::unique_ptr<BFMatcher> create(int normType = NORM_L2, bool crossCheck = false) {
        return std::make_unique<BFMatcher>(normType, crossCheck);
    }

private:
    int normType_;
    bool crossCheck_;

    float computeDistance(
        const std::vector<uint8_t>& a,
        const std::vector<uint8_t>& b
    ) {
        if (a.size() != b.size()) {
            return std::numeric_limits<float>::infinity();
        }
        
        if (normType_ == NORM_HAMMING || normType_ == NORM_HAMMING2) {
            // Hamming distance
            int dist = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                uint8_t xored = a[i] ^ b[i];
                // Count set bits
                while (xored) {
                    dist += xored & 1;
                    xored >>= 1;
                }
            }
            return static_cast<float>(dist);
        } else if (normType_ == NORM_L1) {
            float dist = 0.0f;
            for (size_t i = 0; i < a.size(); ++i) {
                dist += std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
            }
            return dist;
        } else {
            // L2 distance
            float dist = 0.0f;
            for (size_t i = 0; i < a.size(); ++i) {
                float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
                dist += diff * diff;
            }
            return std::sqrt(dist);
        }
    }
};

inline std::unique_ptr<BFMatcher> BFMatcher_create(int normType = NORM_L2, bool crossCheck = false) {
    return BFMatcher::create(normType, crossCheck);
}

// ============================================================================
// FLANN-Based Matcher (Simplified)
// ============================================================================

/**
 * @brief FLANN-based Matcher (simplified, uses brute-force internally)
 */
class FlannBasedMatcher : public DescriptorMatcher {
public:
    FlannBasedMatcher() : bf_(NORM_L2, false) {}

    std::vector<DMatch> match(
        const std::vector<std::vector<uint8_t>>& queryDescriptors,
        const std::vector<std::vector<uint8_t>>& trainDescriptors
    ) override {
        return bf_.match(queryDescriptors, trainDescriptors);
    }

    std::vector<std::vector<DMatch>> knnMatch(
        const std::vector<std::vector<uint8_t>>& queryDescriptors,
        const std::vector<std::vector<uint8_t>>& trainDescriptors,
        int k
    ) override {
        return bf_.knnMatch(queryDescriptors, trainDescriptors, k);
    }

    std::vector<std::vector<DMatch>> radiusMatch(
        const std::vector<std::vector<uint8_t>>& queryDescriptors,
        const std::vector<std::vector<uint8_t>>& trainDescriptors,
        float maxDistance
    ) override {
        return bf_.radiusMatch(queryDescriptors, trainDescriptors, maxDistance);
    }

    static std::unique_ptr<FlannBasedMatcher> create() {
        return std::make_unique<FlannBasedMatcher>();
    }

private:
    BFMatcher bf_;
};

inline std::unique_ptr<FlannBasedMatcher> FlannBasedMatcher_create() {
    return FlannBasedMatcher::create();
}

// ============================================================================
// Drawing Functions
// ============================================================================

/**
 * @brief Draw keypoints on image
 */
inline void drawKeypoints(
    float* image, int width, int height, int channels,
    const std::vector<KeyPoint>& keypoints,
    const float color[3] = nullptr,
    int flags = DRAW_MATCHES_FLAGS_DEFAULT
) {
    float defaultColor[3] = {0.0f, 255.0f, 0.0f};
    const float* c = color ? color : defaultColor;
    
    bool richKeypoints = (flags & DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) != 0;
    
    for (const auto& kp : keypoints) {
        int cx = static_cast<int>(kp.x);
        int cy = static_cast<int>(kp.y);
        int radius = richKeypoints ? static_cast<int>(kp.size / 2) : 3;
        
        // Draw circle
        for (int angle = 0; angle < 360; angle += 10) {
            float rad = angle * 3.14159265f / 180.0f;
            int px = cx + static_cast<int>(radius * std::cos(rad));
            int py = cy + static_cast<int>(radius * std::sin(rad));
            
            if (px >= 0 && px < width && py >= 0 && py < height) {
                for (int ch = 0; ch < std::min(channels, 3); ++ch) {
                    image[(py * width + px) * channels + ch] = c[ch];
                }
            }
        }
        
        // Draw orientation line if rich
        if (richKeypoints && kp.angle >= 0) {
            float rad = kp.angle * 3.14159265f / 180.0f;
            int ex = cx + static_cast<int>(radius * std::cos(rad));
            int ey = cy + static_cast<int>(radius * std::sin(rad));
            
            // Simple line drawing
            int steps = std::max(std::abs(ex - cx), std::abs(ey - cy));
            for (int s = 0; s <= steps; ++s) {
                int lx = cx + (ex - cx) * s / (steps + 1);
                int ly = cy + (ey - cy) * s / (steps + 1);
                
                if (lx >= 0 && lx < width && ly >= 0 && ly < height) {
                    for (int ch = 0; ch < std::min(channels, 3); ++ch) {
                        image[(ly * width + lx) * channels + ch] = c[ch];
                    }
                }
            }
        }
    }
}

/**
 * @brief Draw matches between two images
 */
inline std::vector<float> drawMatches(
    const float* img1, int w1, int h1,
    const float* img2, int w2, int h2,
    int channels,
    const std::vector<KeyPoint>& keypoints1,
    const std::vector<KeyPoint>& keypoints2,
    const std::vector<DMatch>& matches,
    int& outWidth, int& outHeight
) {
    // Side-by-side output
    outWidth = w1 + w2;
    outHeight = std::max(h1, h2);
    
    std::vector<float> output(outWidth * outHeight * channels, 0.0f);
    
    // Copy first image
    for (int y = 0; y < h1; ++y) {
        for (int x = 0; x < w1; ++x) {
            for (int c = 0; c < channels; ++c) {
                output[(y * outWidth + x) * channels + c] = 
                    img1[(y * w1 + x) * channels + c];
            }
        }
    }
    
    // Copy second image
    for (int y = 0; y < h2; ++y) {
        for (int x = 0; x < w2; ++x) {
            for (int c = 0; c < channels; ++c) {
                output[(y * outWidth + (w1 + x)) * channels + c] = 
                    img2[(y * w2 + x) * channels + c];
            }
        }
    }
    
    // Draw matches
    float matchColor[3] = {0.0f, 255.0f, 0.0f};
    
    for (const auto& m : matches) {
        if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(keypoints1.size())) continue;
        if (m.trainIdx < 0 || m.trainIdx >= static_cast<int>(keypoints2.size())) continue;
        
        const auto& kp1 = keypoints1[m.queryIdx];
        const auto& kp2 = keypoints2[m.trainIdx];
        
        int x1 = static_cast<int>(kp1.x);
        int y1 = static_cast<int>(kp1.y);
        int x2 = static_cast<int>(kp2.x) + w1;
        int y2 = static_cast<int>(kp2.y);
        
        // Draw line
        int steps = std::max(std::abs(x2 - x1), std::abs(y2 - y1));
        for (int s = 0; s <= steps; ++s) {
            int lx = x1 + (x2 - x1) * s / (steps + 1);
            int ly = y1 + (y2 - y1) * s / (steps + 1);
            
            if (lx >= 0 && lx < outWidth && ly >= 0 && ly < outHeight) {
                for (int c = 0; c < std::min(channels, 3); ++c) {
                    output[(ly * outWidth + lx) * channels + c] = matchColor[c];
                }
            }
        }
    }
    
    // Draw keypoints
    drawKeypoints(output.data(), outWidth, outHeight, channels, keypoints1);
    
    // Shift keypoints2 for second image
    std::vector<KeyPoint> shifted = keypoints2;
    for (auto& kp : shifted) {
        kp.x += w1;
    }
    drawKeypoints(output.data(), outWidth, outHeight, channels, shifted);
    
    return output;
}

} // namespace features
} // namespace neurova

#endif // NEUROVA_FEATURES_MATCHING_HPP
