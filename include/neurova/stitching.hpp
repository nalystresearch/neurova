// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file stitching.hpp
 * @brief Image stitching for panorama creation
 */

#ifndef NEUROVA_STITCHING_HPP
#define NEUROVA_STITCHING_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace neurova {
namespace stitching {

// ============================================================================
// Enums and Status Codes
// ============================================================================

enum class StitcherMode {
    PANORAMA = 0,
    SCANS = 1
};

enum class StitcherStatus {
    OK = 0,
    ERR_NEED_MORE_IMGS = 1,
    ERR_HOMOGRAPHY_EST_FAIL = 2,
    ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
};

// ============================================================================
// Keypoint and Match Structures
// ============================================================================

struct StitchKeypoint {
    float x, y;
    float response;
    
    StitchKeypoint() : x(0), y(0), response(0) {}
    StitchKeypoint(float x_, float y_, float r = 0) : x(x_), y(y_), response(r) {}
};

struct StitchMatch {
    int queryIdx;
    int trainIdx;
    float distance;
    
    StitchMatch() : queryIdx(-1), trainIdx(-1), distance(0) {}
    StitchMatch(int q, int t, float d) : queryIdx(q), trainIdx(t), distance(d) {}
};

// ============================================================================
// Feature Detection
// ============================================================================

/**
 * @brief Detect Harris corners in an image
 */
inline std::vector<StitchKeypoint> detectHarrisCorners(
    const uint8_t* gray, int width, int height,
    float threshold = 0.01f, int maxKeypoints = 500) {
    
    std::vector<StitchKeypoint> keypoints;
    
    if (width < 10 || height < 10) return keypoints;
    
    // Compute gradients
    std::vector<float> Ix(width * height, 0);
    std::vector<float> Iy(width * height, 0);
    
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            Ix[idx] = static_cast<float>(gray[idx + 1]) - static_cast<float>(gray[idx - 1]);
            Iy[idx] = static_cast<float>(gray[idx + width]) - static_cast<float>(gray[idx - width]);
        }
    }
    
    // Compute structure tensor elements
    std::vector<float> Ixx(width * height), Iyy(width * height), Ixy(width * height);
    for (size_t i = 0; i < Ix.size(); ++i) {
        Ixx[i] = Ix[i] * Ix[i];
        Iyy[i] = Iy[i] * Iy[i];
        Ixy[i] = Ix[i] * Iy[i];
    }
    
    // Box filter (3x3)
    auto boxFilter = [&](const std::vector<float>& src, std::vector<float>& dst) {
        dst.assign(width * height, 0);
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                float sum = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        sum += src[(y + dy) * width + (x + dx)];
                    }
                }
                dst[y * width + x] = sum / 9.0f;
            }
        }
    };
    
    std::vector<float> Sxx(width * height), Syy(width * height), Sxy(width * height);
    boxFilter(Ixx, Sxx);
    boxFilter(Iyy, Syy);
    boxFilter(Ixy, Sxy);
    
    // Compute Harris response
    std::vector<float> response(width * height, 0);
    float maxResponse = 0;
    
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            float det = Sxx[idx] * Syy[idx] - Sxy[idx] * Sxy[idx];
            float trace = Sxx[idx] + Syy[idx];
            response[idx] = det - 0.04f * trace * trace;
            maxResponse = std::max(maxResponse, response[idx]);
        }
    }
    
    // Find local maxima
    float thresh = threshold * maxResponse;
    
    for (int y = 5; y < height - 5; ++y) {
        for (int x = 5; x < width - 5; ++x) {
            int idx = y * width + x;
            if (response[idx] <= thresh) continue;
            
            bool isMax = true;
            for (int dy = -2; dy <= 2 && isMax; ++dy) {
                for (int dx = -2; dx <= 2 && isMax; ++dx) {
                    if (dy == 0 && dx == 0) continue;
                    if (response[(y + dy) * width + (x + dx)] >= response[idx]) {
                        isMax = false;
                    }
                }
            }
            
            if (isMax) {
                keypoints.emplace_back(static_cast<float>(x), static_cast<float>(y), response[idx]);
            }
        }
    }
    
    // Sort by response and limit
    std::sort(keypoints.begin(), keypoints.end(),
              [](const StitchKeypoint& a, const StitchKeypoint& b) {
                  return a.response > b.response;
              });
    
    if (static_cast<int>(keypoints.size()) > maxKeypoints) {
        keypoints.resize(maxKeypoints);
    }
    
    return keypoints;
}

/**
 * @brief Compute simple patch descriptors
 */
inline std::vector<std::vector<float>> computeDescriptors(
    const uint8_t* gray, int width, int height,
    const std::vector<StitchKeypoint>& keypoints, int patchSize = 16) {
    
    std::vector<std::vector<float>> descriptors;
    int halfPatch = patchSize / 2;
    
    for (const auto& kp : keypoints) {
        int x = static_cast<int>(kp.x);
        int y = static_cast<int>(kp.y);
        
        std::vector<float> desc(patchSize * patchSize, 0);
        
        if (x >= halfPatch && x < width - halfPatch && 
            y >= halfPatch && y < height - halfPatch) {
            
            float sum = 0, sumSq = 0;
            int idx = 0;
            
            for (int dy = -halfPatch; dy < halfPatch; ++dy) {
                for (int dx = -halfPatch; dx < halfPatch; ++dx) {
                    float val = static_cast<float>(gray[(y + dy) * width + (x + dx)]);
                    desc[idx++] = val;
                    sum += val;
                    sumSq += val * val;
                }
            }
            
            // Normalize
            float mean = sum / desc.size();
            float stddev = std::sqrt(sumSq / desc.size() - mean * mean + 1e-10f);
            
            for (float& v : desc) {
                v = (v - mean) / stddev;
            }
        }
        
        descriptors.push_back(desc);
    }
    
    return descriptors;
}

/**
 * @brief Match descriptors between two images
 */
inline std::vector<StitchMatch> matchDescriptors(
    const std::vector<std::vector<float>>& desc1,
    const std::vector<std::vector<float>>& desc2,
    float ratioThresh = 0.75f) {
    
    std::vector<StitchMatch> matches;
    
    for (size_t i = 0; i < desc1.size(); ++i) {
        float best = std::numeric_limits<float>::max();
        float second = std::numeric_limits<float>::max();
        int bestIdx = -1;
        
        for (size_t j = 0; j < desc2.size(); ++j) {
            float dist = 0;
            for (size_t k = 0; k < desc1[i].size() && k < desc2[j].size(); ++k) {
                float diff = desc1[i][k] - desc2[j][k];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);
            
            if (dist < best) {
                second = best;
                best = dist;
                bestIdx = static_cast<int>(j);
            } else if (dist < second) {
                second = dist;
            }
        }
        
        // Ratio test
        if (bestIdx >= 0 && best < ratioThresh * second) {
            matches.emplace_back(static_cast<int>(i), bestIdx, best);
        }
    }
    
    return matches;
}

// ============================================================================
// Homography Estimation
// ============================================================================

/**
 * @brief Estimate homography using DLT (Direct Linear Transform)
 */
inline bool estimateHomography(
    const std::vector<std::pair<float, float>>& srcPts,
    const std::vector<std::pair<float, float>>& dstPts,
    float H[9]) {
    
    if (srcPts.size() < 4 || srcPts.size() != dstPts.size()) {
        return false;
    }
    
    int n = static_cast<int>(srcPts.size());
    
    // Build system Ah = 0
    std::vector<float> A(2 * n * 9, 0);
    
    for (int i = 0; i < n; ++i) {
        float x = srcPts[i].first;
        float y = srcPts[i].second;
        float xp = dstPts[i].first;
        float yp = dstPts[i].second;
        
        int row1 = 2 * i;
        int row2 = 2 * i + 1;
        
        A[row1 * 9 + 0] = -x;
        A[row1 * 9 + 1] = -y;
        A[row1 * 9 + 2] = -1;
        A[row1 * 9 + 6] = x * xp;
        A[row1 * 9 + 7] = y * xp;
        A[row1 * 9 + 8] = xp;
        
        A[row2 * 9 + 3] = -x;
        A[row2 * 9 + 4] = -y;
        A[row2 * 9 + 5] = -1;
        A[row2 * 9 + 6] = x * yp;
        A[row2 * 9 + 7] = y * yp;
        A[row2 * 9 + 8] = yp;
    }
    
    // Solve using simplified approach (least squares)
    // For proper implementation, use SVD
    // This is a simplified version
    
    // Compute A^T * A
    std::vector<float> ATA(81, 0);
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            float sum = 0;
            for (int k = 0; k < 2 * n; ++k) {
                sum += A[k * 9 + i] * A[k * 9 + j];
            }
            ATA[i * 9 + j] = sum;
        }
    }
    
    // Power iteration to find smallest eigenvector
    std::vector<float> h(9, 1.0f / 3.0f);
    
    for (int iter = 0; iter < 100; ++iter) {
        std::vector<float> hNew(9, 0);
        
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                hNew[i] += ATA[i * 9 + j] * h[j];
            }
        }
        
        // Normalize
        float norm = 0;
        for (float v : hNew) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 1e-10f) {
            for (float& v : hNew) v /= norm;
        }
        
        h = hNew;
    }
    
    // Copy result
    for (int i = 0; i < 9; ++i) {
        H[i] = h[i];
    }
    
    // Normalize so H[8] = 1
    if (std::abs(H[8]) > 1e-10f) {
        for (int i = 0; i < 9; ++i) {
            H[i] /= H[8];
        }
    }
    
    return true;
}

/**
 * @brief RANSAC homography estimation
 */
inline bool estimateHomographyRANSAC(
    const std::vector<StitchKeypoint>& kp1,
    const std::vector<StitchKeypoint>& kp2,
    const std::vector<StitchMatch>& matches,
    float H[9],
    float ransacThresh = 3.0f,
    int maxIters = 1000) {
    
    if (matches.size() < 4) return false;
    
    int bestInliers = 0;
    float bestH[9] = {1,0,0, 0,1,0, 0,0,1};
    
    unsigned int seed = 12345;
    
    for (int iter = 0; iter < maxIters; ++iter) {
        // Random sample 4 points
        std::vector<int> indices;
        while (indices.size() < 4) {
            seed = seed * 1103515245 + 12345;
            int idx = (seed >> 16) % matches.size();
            
            bool found = false;
            for (int i : indices) if (i == idx) found = true;
            if (!found) indices.push_back(idx);
        }
        
        std::vector<std::pair<float, float>> srcPts, dstPts;
        for (int idx : indices) {
            const auto& m = matches[idx];
            srcPts.emplace_back(kp1[m.queryIdx].x, kp1[m.queryIdx].y);
            dstPts.emplace_back(kp2[m.trainIdx].x, kp2[m.trainIdx].y);
        }
        
        float tempH[9];
        if (!estimateHomography(srcPts, dstPts, tempH)) continue;
        
        // Count inliers
        int inliers = 0;
        for (const auto& m : matches) {
            float x = kp1[m.queryIdx].x;
            float y = kp1[m.queryIdx].y;
            
            float w = tempH[6] * x + tempH[7] * y + tempH[8];
            if (std::abs(w) < 1e-10f) continue;
            
            float xp = (tempH[0] * x + tempH[1] * y + tempH[2]) / w;
            float yp = (tempH[3] * x + tempH[4] * y + tempH[5]) / w;
            
            float dx = xp - kp2[m.trainIdx].x;
            float dy = yp - kp2[m.trainIdx].y;
            float dist = std::sqrt(dx * dx + dy * dy);
            
            if (dist < ransacThresh) inliers++;
        }
        
        if (inliers > bestInliers) {
            bestInliers = inliers;
            for (int i = 0; i < 9; ++i) bestH[i] = tempH[i];
        }
    }
    
    for (int i = 0; i < 9; ++i) H[i] = bestH[i];
    
    return bestInliers >= 4;
}

// ============================================================================
// Image Warping and Blending
// ============================================================================

/**
 * @brief Warp image using homography
 */
inline void warpPerspective(
    const uint8_t* src, int srcW, int srcH, int srcC,
    uint8_t* dst, int dstW, int dstH,
    const float H[9]) {
    
    // Compute inverse homography
    float det = H[0] * (H[4] * H[8] - H[5] * H[7]) -
                H[1] * (H[3] * H[8] - H[5] * H[6]) +
                H[2] * (H[3] * H[7] - H[4] * H[6]);
    
    if (std::abs(det) < 1e-10f) return;
    
    float Hinv[9];
    Hinv[0] = (H[4] * H[8] - H[5] * H[7]) / det;
    Hinv[1] = (H[2] * H[7] - H[1] * H[8]) / det;
    Hinv[2] = (H[1] * H[5] - H[2] * H[4]) / det;
    Hinv[3] = (H[5] * H[6] - H[3] * H[8]) / det;
    Hinv[4] = (H[0] * H[8] - H[2] * H[6]) / det;
    Hinv[5] = (H[2] * H[3] - H[0] * H[5]) / det;
    Hinv[6] = (H[3] * H[7] - H[4] * H[6]) / det;
    Hinv[7] = (H[1] * H[6] - H[0] * H[7]) / det;
    Hinv[8] = (H[0] * H[4] - H[1] * H[3]) / det;
    
    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            float w = Hinv[6] * x + Hinv[7] * y + Hinv[8];
            if (std::abs(w) < 1e-10f) continue;
            
            float srcX = (Hinv[0] * x + Hinv[1] * y + Hinv[2]) / w;
            float srcY = (Hinv[3] * x + Hinv[4] * y + Hinv[5]) / w;
            
            if (srcX >= 0 && srcX < srcW - 1 && srcY >= 0 && srcY < srcH - 1) {
                int x0 = static_cast<int>(srcX);
                int y0 = static_cast<int>(srcY);
                float fx = srcX - x0;
                float fy = srcY - y0;
                
                for (int c = 0; c < srcC; ++c) {
                    float v00 = src[(y0 * srcW + x0) * srcC + c];
                    float v01 = src[(y0 * srcW + x0 + 1) * srcC + c];
                    float v10 = src[((y0 + 1) * srcW + x0) * srcC + c];
                    float v11 = src[((y0 + 1) * srcW + x0 + 1) * srcC + c];
                    
                    float val = v00 * (1 - fx) * (1 - fy) +
                               v01 * fx * (1 - fy) +
                               v10 * (1 - fx) * fy +
                               v11 * fx * fy;
                    
                    dst[(y * dstW + x) * srcC + c] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
                }
            }
        }
    }
}

/**
 * @brief Blend two images with linear blending
 */
inline void blendImages(
    const uint8_t* img1, const uint8_t* img2,
    uint8_t* result, int width, int height, int channels,
    float blendWidth = 50.0f) {
    
    // Find overlap region
    std::vector<bool> mask1(width * height, false);
    std::vector<bool> mask2(width * height, false);
    
    for (int i = 0; i < width * height; ++i) {
        bool hasValue1 = false, hasValue2 = false;
        for (int c = 0; c < channels; ++c) {
            if (img1[i * channels + c] > 0) hasValue1 = true;
            if (img2[i * channels + c] > 0) hasValue2 = true;
        }
        mask1[i] = hasValue1;
        mask2[i] = hasValue2;
    }
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            
            bool in1 = mask1[idx];
            bool in2 = mask2[idx];
            
            for (int c = 0; c < channels; ++c) {
                if (in1 && in2) {
                    // Blend in overlap
                    float alpha = 0.5f;
                    result[idx * channels + c] = static_cast<uint8_t>(
                        alpha * img1[idx * channels + c] +
                        (1 - alpha) * img2[idx * channels + c]);
                } else if (in1) {
                    result[idx * channels + c] = img1[idx * channels + c];
                } else if (in2) {
                    result[idx * channels + c] = img2[idx * channels + c];
                } else {
                    result[idx * channels + c] = 0;
                }
            }
        }
    }
}

// ============================================================================
// Stitcher Class
// ============================================================================

/**
 * @brief High-level image stitcher
 */
class Stitcher {
public:
    Stitcher(StitcherMode mode = StitcherMode::PANORAMA) 
        : mode_(mode), registrationResol_(0.6f), 
          seamEstimationResol_(0.1f), compositingResol_(-1.0f),
          panoConfidenceThresh_(1.0f) {}
    
    static Stitcher create(StitcherMode mode = StitcherMode::PANORAMA) {
        return Stitcher(mode);
    }
    
    /**
     * @brief Stitch multiple images into a panorama
     */
    StitcherStatus stitch(
        const std::vector<const uint8_t*>& images,
        const std::vector<int>& widths,
        const std::vector<int>& heights,
        int channels,
        std::vector<uint8_t>& result,
        int& resultWidth, int& resultHeight) {
        
        if (images.size() < 2) {
            return StitcherStatus::ERR_NEED_MORE_IMGS;
        }
        
        // Convert to grayscale for feature detection
        std::vector<std::vector<uint8_t>> grays;
        for (size_t i = 0; i < images.size(); ++i) {
            std::vector<uint8_t> gray(widths[i] * heights[i]);
            for (int j = 0; j < widths[i] * heights[i]; ++j) {
                if (channels >= 3) {
                    gray[j] = static_cast<uint8_t>(
                        0.299f * images[i][j * channels] +
                        0.587f * images[i][j * channels + 1] +
                        0.114f * images[i][j * channels + 2]);
                } else {
                    gray[j] = images[i][j * channels];
                }
            }
            grays.push_back(gray);
        }
        
        // Detect features
        std::vector<std::vector<StitchKeypoint>> allKeypoints;
        std::vector<std::vector<std::vector<float>>> allDescriptors;
        
        for (size_t i = 0; i < images.size(); ++i) {
            auto kps = detectHarrisCorners(grays[i].data(), widths[i], heights[i]);
            auto descs = computeDescriptors(grays[i].data(), widths[i], heights[i], kps);
            allKeypoints.push_back(kps);
            allDescriptors.push_back(descs);
        }
        
        // Estimate homographies (chain from first image)
        std::vector<std::array<float, 9>> homographies(images.size());
        homographies[0] = {1,0,0, 0,1,0, 0,0,1};
        
        for (size_t i = 1; i < images.size(); ++i) {
            auto matches = matchDescriptors(allDescriptors[i-1], allDescriptors[i]);
            
            if (matches.size() < 4) {
                return StitcherStatus::ERR_HOMOGRAPHY_EST_FAIL;
            }
            
            float H[9];
            if (!estimateHomographyRANSAC(allKeypoints[i-1], allKeypoints[i], matches, H)) {
                return StitcherStatus::ERR_HOMOGRAPHY_EST_FAIL;
            }
            
            // Chain homography
            float prevH[9];
            for (int j = 0; j < 9; ++j) prevH[j] = homographies[i-1][j];
            
            // H_new = H_prev * H
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    float sum = 0;
                    for (int k = 0; k < 3; ++k) {
                        sum += prevH[row * 3 + k] * H[k * 3 + col];
                    }
                    homographies[i][row * 3 + col] = sum;
                }
            }
        }
        
        // Compute output bounds
        float minX = 0, maxX = static_cast<float>(widths[0]);
        float minY = 0, maxY = static_cast<float>(heights[0]);
        
        for (size_t i = 1; i < images.size(); ++i) {
            float corners[4][2] = {
                {0, 0}, 
                {static_cast<float>(widths[i]), 0},
                {0, static_cast<float>(heights[i])},
                {static_cast<float>(widths[i]), static_cast<float>(heights[i])}
            };
            
            for (auto& c : corners) {
                const auto& H = homographies[i];
                float w = H[6] * c[0] + H[7] * c[1] + H[8];
                if (std::abs(w) > 1e-10f) {
                    float x = (H[0] * c[0] + H[1] * c[1] + H[2]) / w;
                    float y = (H[3] * c[0] + H[4] * c[1] + H[5]) / w;
                    minX = std::min(minX, x);
                    maxX = std::max(maxX, x);
                    minY = std::min(minY, y);
                    maxY = std::max(maxY, y);
                }
            }
        }
        
        resultWidth = static_cast<int>(maxX - minX + 1);
        resultHeight = static_cast<int>(maxY - minY + 1);
        
        result.resize(resultWidth * resultHeight * channels, 0);
        
        // Warp and blend each image
        for (size_t i = 0; i < images.size(); ++i) {
            std::vector<uint8_t> warped(resultWidth * resultHeight * channels, 0);
            
            // Adjust homography for offset
            std::array<float, 9> H = homographies[i];
            H[2] -= minX;
            H[5] -= minY;
            
            warpPerspective(images[i], widths[i], heights[i], channels,
                          warped.data(), resultWidth, resultHeight, H.data());
            
            blendImages(result.data(), warped.data(), result.data(),
                       resultWidth, resultHeight, channels);
        }
        
        return StitcherStatus::OK;
    }
    
    // Getters/setters
    void setRegistrationResol(float resol) { registrationResol_ = resol; }
    float registrationResol() const { return registrationResol_; }
    
    void setSeamEstimationResol(float resol) { seamEstimationResol_ = resol; }
    float seamEstimationResol() const { return seamEstimationResol_; }
    
    void setCompositingResol(float resol) { compositingResol_ = resol; }
    float compositingResol() const { return compositingResol_; }
    
    void setPanoConfidenceThresh(float thresh) { panoConfidenceThresh_ = thresh; }
    float panoConfidenceThresh() const { return panoConfidenceThresh_; }
    
private:
    StitcherMode mode_;
    float registrationResol_;
    float seamEstimationResol_;
    float compositingResol_;
    float panoConfidenceThresh_;
};

} // namespace stitching
} // namespace neurova

#endif // NEUROVA_STITCHING_HPP
