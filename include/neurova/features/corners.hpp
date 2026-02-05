// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file corners.hpp
 * @brief Corner detection algorithms
 */

#ifndef NEUROVA_FEATURES_CORNERS_HPP
#define NEUROVA_FEATURES_CORNERS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "keypoint.hpp"

namespace neurova {
namespace features {

// ============================================================================
// Sobel Gradients
// ============================================================================

/**
 * @brief Compute Sobel gradients
 */
inline void sobel(
    const float* gray, int width, int height,
    std::vector<float>& gx, std::vector<float>& gy
) {
    gx.resize(width * height, 0.0f);
    gy.resize(width * height, 0.0f);
    
    // Sobel kernels
    const float kx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const float ky[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float sx = 0.0f, sy = 0.0f;
            int k = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    float val = gray[(y + dy) * width + (x + dx)];
                    sx += val * kx[k];
                    sy += val * ky[k];
                    ++k;
                }
            }
            gx[y * width + x] = sx;
            gy[y * width + x] = sy;
        }
    }
}

/**
 * @brief Apply box blur
 */
inline void boxBlur(
    const float* src, int width, int height,
    float* dst, int ksize
) {
    int radius = ksize / 2;
    float scale = 1.0f / (ksize * ksize);
    
    std::vector<float> temp(width * height);
    
    // Horizontal pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = std::clamp(x + dx, 0, width - 1);
                sum += src[y * width + nx];
            }
            temp[y * width + x] = sum;
        }
    }
    
    // Vertical pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int dy = -radius; dy <= radius; ++dy) {
                int ny = std::clamp(y + dy, 0, height - 1);
                sum += temp[ny * width + x];
            }
            dst[y * width + x] = sum * scale;
        }
    }
}

// ============================================================================
// Harris Corner Response
// ============================================================================

/**
 * @brief Compute Harris corner response
 */
inline std::vector<float> harrisResponse(
    const float* gray, int width, int height,
    int blockSize = 3, float k = 0.04f
) {
    std::vector<float> gx, gy;
    sobel(gray, width, height, gx, gy);
    
    // Compute products
    std::vector<float> ixx(width * height);
    std::vector<float> iyy(width * height);
    std::vector<float> ixy(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        ixx[i] = gx[i] * gx[i];
        iyy[i] = gy[i] * gy[i];
        ixy[i] = gx[i] * gy[i];
    }
    
    // Blur products
    std::vector<float> sxx(width * height);
    std::vector<float> syy(width * height);
    std::vector<float> sxy(width * height);
    
    boxBlur(ixx.data(), width, height, sxx.data(), blockSize);
    boxBlur(iyy.data(), width, height, syy.data(), blockSize);
    boxBlur(ixy.data(), width, height, sxy.data(), blockSize);
    
    // Compute response
    std::vector<float> response(width * height);
    for (int i = 0; i < width * height; ++i) {
        float det = sxx[i] * syy[i] - sxy[i] * sxy[i];
        float trace = sxx[i] + syy[i];
        response[i] = det - k * trace * trace;
    }
    
    return response;
}

/**
 * @brief Compute Shi-Tomasi (minimum eigenvalue) response
 */
inline std::vector<float> shiTomasiResponse(
    const float* gray, int width, int height,
    int blockSize = 3
) {
    std::vector<float> gx, gy;
    sobel(gray, width, height, gx, gy);
    
    std::vector<float> ixx(width * height);
    std::vector<float> iyy(width * height);
    std::vector<float> ixy(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        ixx[i] = gx[i] * gx[i];
        iyy[i] = gy[i] * gy[i];
        ixy[i] = gx[i] * gy[i];
    }
    
    std::vector<float> sxx(width * height);
    std::vector<float> syy(width * height);
    std::vector<float> sxy(width * height);
    
    boxBlur(ixx.data(), width, height, sxx.data(), blockSize);
    boxBlur(iyy.data(), width, height, syy.data(), blockSize);
    boxBlur(ixy.data(), width, height, sxy.data(), blockSize);
    
    // Min eigenvalue
    std::vector<float> response(width * height);
    for (int i = 0; i < width * height; ++i) {
        float a = sxx[i], b = sxy[i], c = syy[i];
        float t = a + c;
        float d = std::max(0.0f, (a - c) * (a - c) + 4.0f * b * b);
        float sqrt_d = std::sqrt(d);
        float lam1 = 0.5f * (t + sqrt_d);
        float lam2 = 0.5f * (t - sqrt_d);
        response[i] = std::min(lam1, lam2);
    }
    
    return response;
}

// ============================================================================
// Non-Maximum Suppression
// ============================================================================

/**
 * @brief Non-maximum suppression for corner response
 */
inline std::vector<std::pair<int, int>> nonMaxSuppression(
    const std::vector<float>& response,
    int width, int height,
    float qualityLevel, int minDistance, int maxCorners
) {
    // Find maximum response
    float maxResp = *std::max_element(response.begin(), response.end());
    if (maxResp <= 0.0f) return {};
    
    float threshold = qualityLevel * maxResp;
    
    // Find local maxima
    std::vector<std::tuple<float, int, int>> candidates;
    
    for (int y = minDistance; y < height - minDistance; ++y) {
        for (int x = minDistance; x < width - minDistance; ++x) {
            float r = response[y * width + x];
            if (r < threshold) continue;
            
            // Check if local maximum
            bool isMax = true;
            for (int dy = -1; dy <= 1 && isMax; ++dy) {
                for (int dx = -1; dx <= 1 && isMax; ++dx) {
                    if (dy == 0 && dx == 0) continue;
                    if (response[(y + dy) * width + (x + dx)] >= r) {
                        isMax = false;
                    }
                }
            }
            
            if (isMax) {
                candidates.emplace_back(r, x, y);
            }
        }
    }
    
    // Sort by response
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                  return std::get<0>(a) > std::get<0>(b);
              });
    
    // Apply minimum distance constraint
    std::vector<std::pair<int, int>> corners;
    std::vector<bool> used(width * height, false);
    
    for (const auto& [resp, cx, cy] : candidates) {
        if (maxCorners > 0 && static_cast<int>(corners.size()) >= maxCorners) break;
        
        bool tooClose = false;
        for (int dy = -minDistance; dy <= minDistance && !tooClose; ++dy) {
            for (int dx = -minDistance; dx <= minDistance && !tooClose; ++dx) {
                int nx = cx + dx;
                int ny = cy + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    if (used[ny * width + nx]) {
                        tooClose = true;
                    }
                }
            }
        }
        
        if (!tooClose) {
            corners.emplace_back(cx, cy);
            used[cy * width + cx] = true;
        }
    }
    
    return corners;
}

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Detect corners in image
 * 
 * @param gray Grayscale image
 * @param width Image width
 * @param height Image height
 * @param method "harris" or "shi_tomasi"
 * @param maxCorners Maximum number of corners (0 = unlimited)
 * @param qualityLevel Quality threshold (0.01)
 * @param minDistance Minimum distance between corners
 * @param blockSize Block size for gradient computation
 * @param k Harris detector parameter
 * @return Vector of (x, y) corner coordinates
 */
inline std::vector<std::pair<int, int>> detectCorners(
    const float* gray, int width, int height,
    const std::string& method = "harris",
    int maxCorners = 200,
    float qualityLevel = 0.01f,
    int minDistance = 5,
    int blockSize = 3,
    float k = 0.04f
) {
    std::vector<float> response;
    
    if (method == "harris") {
        response = harrisResponse(gray, width, height, blockSize, k);
    } else {
        response = shiTomasiResponse(gray, width, height, blockSize);
    }
    
    return nonMaxSuppression(response, width, height, 
                            qualityLevel, minDistance, maxCorners);
}

/**
 * @brief Detect keypoints using corner detection
 */
inline std::vector<KeyPoint> detectKeypoints(
    const float* gray, int width, int height,
    const std::string& method = "harris",
    int maxKeypoints = 500,
    float qualityLevel = 0.01f,
    int minDistance = 5
) {
    auto corners = detectCorners(gray, width, height, method,
                                 maxKeypoints, qualityLevel, minDistance);
    
    std::vector<KeyPoint> keypoints;
    keypoints.reserve(corners.size());
    
    for (const auto& [x, y] : corners) {
        keypoints.emplace_back(static_cast<float>(x), static_cast<float>(y), 1.0f);
    }
    
    return keypoints;
}

} // namespace features
} // namespace neurova

#endif // NEUROVA_FEATURES_CORNERS_HPP
