// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file video/optflow.hpp
 * @brief Optical flow algorithms
 */

#ifndef NEUROVA_VIDEO_OPTFLOW_HPP
#define NEUROVA_VIDEO_OPTFLOW_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace neurova {
namespace video {

// ============================================================================
// Optical Flow Flags
// ============================================================================

enum OpticalFlowFlags {
    OPTFLOW_USE_INITIAL_FLOW = 4,
    OPTFLOW_LK_GET_MIN_EIGENVALS = 8,
    OPTFLOW_FARNEBACK_GAUSSIAN = 256
};

// ============================================================================
// Pyramid Building
// ============================================================================

namespace detail {

inline std::vector<std::vector<float>> buildPyramid(const uint8_t* img, int w, int h, int maxLevel) {
    std::vector<std::vector<float>> pyramid;
    
    // First level - convert to float
    std::vector<float> level0(w * h);
    for (int i = 0; i < w * h; ++i) {
        level0[i] = static_cast<float>(img[i]);
    }
    pyramid.push_back(std::move(level0));
    
    int currW = w, currH = h;
    
    for (int l = 0; l < maxLevel; ++l) {
        int newW = currW / 2;
        int newH = currH / 2;
        if (newW < 5 || newH < 5) break;
        
        const auto& prev = pyramid.back();
        std::vector<float> next(newW * newH);
        
        for (int y = 0; y < newH; ++y) {
            for (int x = 0; x < newW; ++x) {
                float sum = 0;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        sum += prev[(y * 2 + dy) * currW + (x * 2 + dx)];
                    }
                }
                next[y * newW + x] = sum / 4.0f;
            }
        }
        
        pyramid.push_back(std::move(next));
        currW = newW;
        currH = newH;
    }
    
    return pyramid;
}

inline int pyramidWidth(int origW, int level) {
    return origW >> level;
}

inline int pyramidHeight(int origH, int level) {
    return origH >> level;
}

} // namespace detail

// ============================================================================
// Lucas-Kanade Optical Flow (Sparse)
// ============================================================================

struct LKResult {
    std::vector<float> nextPtsX;
    std::vector<float> nextPtsY;
    std::vector<uint8_t> status;
    std::vector<float> error;
};

/**
 * @brief Calculate sparse optical flow using Lucas-Kanade method with pyramids
 */
inline LKResult calcOpticalFlowPyrLK(
    const uint8_t* prevImg, const uint8_t* nextImg,
    int w, int h,
    const float* prevPtsX, const float* prevPtsY, int numPts,
    int winSizeW = 21, int winSizeH = 21,
    int maxLevel = 3,
    int maxIter = 30, float epsilon = 0.01f,
    float minEigThreshold = 1e-4f)
{
    LKResult result;
    result.nextPtsX.resize(numPts);
    result.nextPtsY.resize(numPts);
    result.status.resize(numPts, 1);
    result.error.resize(numPts, 0.0f);
    
    // Build pyramids
    auto prevPyr = detail::buildPyramid(prevImg, w, h, maxLevel);
    auto nextPyr = detail::buildPyramid(nextImg, w, h, maxLevel);
    
    int winW = winSizeW / 2;
    int winH = winSizeH / 2;
    
    // Initialize output with input
    for (int i = 0; i < numPts; ++i) {
        result.nextPtsX[i] = prevPtsX[i];
        result.nextPtsY[i] = prevPtsY[i];
    }
    
    int actualLevels = static_cast<int>(prevPyr.size()) - 1;
    
    // Track from coarse to fine
    for (int level = actualLevels; level >= 0; --level) {
        int levelW = detail::pyramidWidth(w, level);
        int levelH = detail::pyramidHeight(h, level);
        float scale = static_cast<float>(1 << level);
        
        const auto& prevLevel = prevPyr[level];
        const auto& nextLevel = nextPyr[level];
        
        for (int i = 0; i < numPts; ++i) {
            if (result.status[i] == 0) continue;
            
            float px = prevPtsX[i] / scale;
            float py = prevPtsY[i] / scale;
            float dx = result.nextPtsX[i] / scale - px;
            float dy = result.nextPtsY[i] / scale - py;
            
            int pxInt = static_cast<int>(px);
            int pyInt = static_cast<int>(py);
            
            // Check bounds
            if (pxInt - winW < 0 || pxInt + winW >= levelW ||
                pyInt - winH < 0 || pyInt + winH >= levelH) {
                result.status[i] = 0;
                continue;
            }
            
            // Compute gradients and structure tensor
            float Ixx = 0, Ixy = 0, Iyy = 0;
            std::vector<float> Ix((2*winW+1) * (2*winH+1));
            std::vector<float> Iy((2*winW+1) * (2*winH+1));
            std::vector<float> templatePatch((2*winW+1) * (2*winH+1));
            
            int idx = 0;
            for (int wy = -winH; wy <= winH; ++wy) {
                for (int wx = -winW; wx <= winW; ++wx) {
                    int px_ = pxInt + wx;
                    int py_ = pyInt + wy;
                    
                    templatePatch[idx] = prevLevel[py_ * levelW + px_];
                    
                    float gx = 0, gy = 0;
                    if (px_ > 0 && px_ < levelW - 1) {
                        gx = (prevLevel[py_ * levelW + px_ + 1] - prevLevel[py_ * levelW + px_ - 1]) / 2.0f;
                    }
                    if (py_ > 0 && py_ < levelH - 1) {
                        gy = (prevLevel[(py_ + 1) * levelW + px_] - prevLevel[(py_ - 1) * levelW + px_]) / 2.0f;
                    }
                    
                    Ix[idx] = gx;
                    Iy[idx] = gy;
                    
                    Ixx += gx * gx;
                    Ixy += gx * gy;
                    Iyy += gy * gy;
                    
                    idx++;
                }
            }
            
            // Check minimum eigenvalue
            float trace = Ixx + Iyy;
            float det = Ixx * Iyy - Ixy * Ixy;
            
            if (trace <= 0) {
                result.status[i] = 0;
                continue;
            }
            
            float discriminant = trace * trace - 4 * det;
            if (discriminant < 0) discriminant = 0;
            float minEig = (trace - std::sqrt(discriminant)) / 2.0f;
            
            if (minEig < minEigThreshold * winSizeW * winSizeH) {
                result.status[i] = 0;
                continue;
            }
            
            // Iterative Lucas-Kanade
            for (int iter = 0; iter < maxIter; ++iter) {
                float nx = px + dx;
                float ny = py + dy;
                
                int nxInt = static_cast<int>(nx);
                int nyInt = static_cast<int>(ny);
                
                if (nxInt - winW < 0 || nxInt + winW >= levelW ||
                    nyInt - winH < 0 || nyInt + winH >= levelH) {
                    result.status[i] = 0;
                    break;
                }
                
                // Compute temporal derivative and flow update
                float bx = 0, by = 0;
                idx = 0;
                for (int wy = -winH; wy <= winH; ++wy) {
                    for (int wx = -winW; wx <= winW; ++wx) {
                        int nx_ = nxInt + wx;
                        int ny_ = nyInt + wy;
                        
                        float It = nextLevel[ny_ * levelW + nx_] - templatePatch[idx];
                        
                        bx += Ix[idx] * It;
                        by += Iy[idx] * It;
                        
                        idx++;
                    }
                }
                
                // Solve 2x2 system
                if (std::abs(det) < 1e-10f) {
                    result.status[i] = 0;
                    break;
                }
                
                float ddx = -(Iyy * bx - Ixy * by) / det;
                float ddy = -(-Ixy * bx + Ixx * by) / det;
                
                dx += ddx;
                dy += ddy;
                
                if (ddx * ddx + ddy * ddy < epsilon * epsilon) {
                    break;
                }
            }
            
            if (result.status[i]) {
                result.nextPtsX[i] = (px + dx) * scale;
                result.nextPtsY[i] = (py + dy) * scale;
            }
        }
    }
    
    return result;
}

// ============================================================================
// Farneback Optical Flow (Dense)
// ============================================================================

/**
 * @brief Compute dense optical flow using Farneback algorithm
 */
inline void calcOpticalFlowFarneback(
    const uint8_t* prev, const uint8_t* next,
    int w, int h,
    float* flowX, float* flowY,
    float pyrScale = 0.5f, int levels = 5,
    int winsize = 13, int iterations = 10,
    int polyN = 5, float polySigma = 1.1f,
    int flags = 0)
{
    // Initialize flow
    bool useInitial = (flags & OPTFLOW_USE_INITIAL_FLOW) != 0;
    if (!useInitial) {
        std::fill(flowX, flowX + w * h, 0.0f);
        std::fill(flowY, flowY + w * h, 0.0f);
    }
    
    // Build pyramids
    std::vector<std::vector<float>> prevPyr, nextPyr;
    prevPyr.push_back(std::vector<float>(prev, prev + w * h));
    nextPyr.push_back(std::vector<float>(next, next + w * h));
    
    // Convert to float
    for (auto& v : prevPyr[0]) v = v;
    for (auto& v : nextPyr[0]) v = v;
    
    int currW = w, currH = h;
    
    for (int l = 1; l < levels; ++l) {
        int newW = static_cast<int>(currW * pyrScale);
        int newH = static_cast<int>(currH * pyrScale);
        if (newW < 10 || newH < 10) break;
        
        std::vector<float> prevDown(newW * newH);
        std::vector<float> nextDown(newW * newH);
        
        // Simple downsampling
        for (int y = 0; y < newH; ++y) {
            for (int x = 0; x < newW; ++x) {
                int srcX = static_cast<int>(x / pyrScale);
                int srcY = static_cast<int>(y / pyrScale);
                srcX = std::min(srcX, currW - 1);
                srcY = std::min(srcY, currH - 1);
                prevDown[y * newW + x] = prevPyr.back()[srcY * currW + srcX];
                nextDown[y * newW + x] = nextPyr.back()[srcY * currW + srcX];
            }
        }
        
        prevPyr.push_back(std::move(prevDown));
        nextPyr.push_back(std::move(nextDown));
        currW = newW;
        currH = newH;
    }
    
    int actualLevels = static_cast<int>(prevPyr.size());
    
    // Process from coarse to fine
    std::vector<float> flowXLevel, flowYLevel;
    
    for (int level = actualLevels - 1; level >= 0; --level) {
        float scale = std::pow(pyrScale, static_cast<float>(level));
        int levelW = static_cast<int>(w * scale);
        int levelH = static_cast<int>(h * scale);
        
        const auto& prevLevel = prevPyr[level];
        const auto& nextLevel = nextPyr[level];
        
        if (level == actualLevels - 1) {
            flowXLevel.resize(levelW * levelH, 0.0f);
            flowYLevel.resize(levelW * levelH, 0.0f);
        } else {
            // Upsample flow from previous level
            int prevLevelW = static_cast<int>(w * std::pow(pyrScale, level + 1));
            int prevLevelH = static_cast<int>(h * std::pow(pyrScale, level + 1));
            
            std::vector<float> newFlowX(levelW * levelH);
            std::vector<float> newFlowY(levelW * levelH);
            
            for (int y = 0; y < levelH; ++y) {
                for (int x = 0; x < levelW; ++x) {
                    int srcX = std::min(static_cast<int>(x * pyrScale), prevLevelW - 1);
                    int srcY = std::min(static_cast<int>(y * pyrScale), prevLevelH - 1);
                    newFlowX[y * levelW + x] = flowXLevel[srcY * prevLevelW + srcX] / pyrScale;
                    newFlowY[y * levelW + x] = flowYLevel[srcY * prevLevelW + srcX] / pyrScale;
                }
            }
            
            flowXLevel = std::move(newFlowX);
            flowYLevel = std::move(newFlowY);
        }
        
        // Compute gradients
        std::vector<float> Ix(levelW * levelH, 0.0f);
        std::vector<float> Iy(levelW * levelH, 0.0f);
        
        for (int y = 1; y < levelH - 1; ++y) {
            for (int x = 1; x < levelW - 1; ++x) {
                Ix[y * levelW + x] = (prevLevel[y * levelW + x + 1] - prevLevel[y * levelW + x - 1]) / 2.0f;
                Iy[y * levelW + x] = (prevLevel[(y + 1) * levelW + x] - prevLevel[(y - 1) * levelW + x]) / 2.0f;
            }
        }
        
        int halfWin = winsize / 2;
        
        // Iterate
        for (int iter = 0; iter < iterations; ++iter) {
            for (int y = halfWin; y < levelH - halfWin; ++y) {
                for (int x = halfWin; x < levelW - halfWin; ++x) {
                    int idx = y * levelW + x;
                    
                    // Warp and compute temporal derivative
                    float wx = x + flowXLevel[idx];
                    float wy = y + flowYLevel[idx];
                    
                    int wx0 = static_cast<int>(wx);
                    int wy0 = static_cast<int>(wy);
                    
                    if (wx0 < 0 || wx0 >= levelW - 1 || wy0 < 0 || wy0 >= levelH - 1) continue;
                    
                    float fx = wx - wx0;
                    float fy = wy - wy0;
                    
                    float warped = nextLevel[wy0 * levelW + wx0] * (1 - fx) * (1 - fy) +
                                  nextLevel[wy0 * levelW + wx0 + 1] * fx * (1 - fy) +
                                  nextLevel[(wy0 + 1) * levelW + wx0] * (1 - fx) * fy +
                                  nextLevel[(wy0 + 1) * levelW + wx0 + 1] * fx * fy;
                    
                    float It = warped - prevLevel[idx];
                    
                    // Local averaging in window
                    float Ixx = 0, Ixy = 0, Iyy = 0, Ixt = 0, Iyt = 0;
                    
                    for (int wy_ = -halfWin; wy_ <= halfWin; ++wy_) {
                        for (int wx_ = -halfWin; wx_ <= halfWin; ++wx_) {
                            int pidx = (y + wy_) * levelW + (x + wx_);
                            Ixx += Ix[pidx] * Ix[pidx];
                            Ixy += Ix[pidx] * Iy[pidx];
                            Iyy += Iy[pidx] * Iy[pidx];
                            Ixt += Ix[pidx] * It;
                            Iyt += Iy[pidx] * It;
                        }
                    }
                    
                    float det = Ixx * Iyy - Ixy * Ixy;
                    if (std::abs(det) > 1e-6f) {
                        float du = -(Iyy * Ixt - Ixy * Iyt) / det;
                        float dv = -(-Ixy * Ixt + Ixx * Iyt) / det;
                        
                        flowXLevel[idx] += du * 0.5f;
                        flowYLevel[idx] += dv * 0.5f;
                    }
                }
            }
        }
    }
    
    // Copy to output (upsample if needed)
    if (flowXLevel.size() == static_cast<size_t>(w * h)) {
        std::copy(flowXLevel.begin(), flowXLevel.end(), flowX);
        std::copy(flowYLevel.begin(), flowYLevel.end(), flowY);
    }
}

// ============================================================================
// Warp Image by Flow
// ============================================================================

inline void warpByFlow(const uint8_t* src, int w, int h,
                      const float* flowX, const float* flowY,
                      uint8_t* dst) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float mapX = x + flowX[y * w + x];
            float mapY = y + flowY[y * w + x];
            
            int x0 = static_cast<int>(std::floor(mapX));
            int y0 = static_cast<int>(std::floor(mapY));
            
            if (x0 < 0 || x0 >= w - 1 || y0 < 0 || y0 >= h - 1) {
                dst[y * w + x] = 0;
                continue;
            }
            
            float fx = mapX - x0;
            float fy = mapY - y0;
            
            float val = src[y0 * w + x0] * (1 - fx) * (1 - fy) +
                       src[y0 * w + x0 + 1] * fx * (1 - fy) +
                       src[(y0 + 1) * w + x0] * (1 - fx) * fy +
                       src[(y0 + 1) * w + x0 + 1] * fx * fy;
            
            dst[y * w + x] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
        }
    }
}

// ============================================================================
// Flow Visualization
// ============================================================================

inline void flowToColor(const float* flowX, const float* flowY, int w, int h,
                       uint8_t* colorImg) {
    // Find max magnitude
    float maxMag = 1.0f;
    for (int i = 0; i < w * h; ++i) {
        float mag = std::sqrt(flowX[i] * flowX[i] + flowY[i] * flowY[i]);
        maxMag = std::max(maxMag, mag);
    }
    
    for (int i = 0; i < w * h; ++i) {
        float fx = flowX[i];
        float fy = flowY[i];
        
        float mag = std::sqrt(fx * fx + fy * fy);
        float angle = std::atan2(fy, fx);
        
        // Convert to HSV then RGB
        float h_ = (angle + 3.14159265f) / (2 * 3.14159265f);
        float s = std::min(1.0f, mag / maxMag);
        float v = 1.0f;
        
        // HSV to RGB
        int hi = static_cast<int>(h_ * 6) % 6;
        float f = h_ * 6 - hi;
        float p = v * (1 - s);
        float q = v * (1 - f * s);
        float t = v * (1 - (1 - f) * s);
        
        float r, g, b;
        switch (hi) {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            default: r = v; g = p; b = q; break;
        }
        
        colorImg[i * 3] = static_cast<uint8_t>(r * 255);
        colorImg[i * 3 + 1] = static_cast<uint8_t>(g * 255);
        colorImg[i * 3 + 2] = static_cast<uint8_t>(b * 255);
    }
}

} // namespace video
} // namespace neurova

#endif // NEUROVA_VIDEO_OPTFLOW_HPP
