// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file transform.hpp
 * @brief Image transformation operations
 */

#ifndef NEUROVA_TRANSFORM_HPP
#define NEUROVA_TRANSFORM_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace neurova {
namespace transform {

// ============================================================================
// Interpolation Modes
// ============================================================================

enum class InterpolationMode {
    NEAREST = 0,
    LINEAR = 1,
    CUBIC = 2,
    LANCZOS = 3,
    AREA = 4
};

enum class BorderMode {
    CONSTANT = 0,
    REPLICATE = 1,
    REFLECT = 2,
    WRAP = 3
};

// ============================================================================
// Affine Transformations
// ============================================================================

/**
 * @brief Get 2x3 rotation matrix
 */
inline void getRotationMatrix2D(float centerX, float centerY, 
                                float angleDegrees, float scale,
                                float M[6]) {
    float angle = angleDegrees * 3.14159265358979f / 180.0f;
    float alpha = scale * std::cos(angle);
    float beta = scale * std::sin(angle);
    
    M[0] = alpha;
    M[1] = beta;
    M[2] = (1.0f - alpha) * centerX - beta * centerY;
    M[3] = -beta;
    M[4] = alpha;
    M[5] = beta * centerX + (1.0f - alpha) * centerY;
}

/**
 * @brief Get affine transform from 3 point correspondences
 */
inline bool getAffineTransform(const float srcPts[6], const float dstPts[6], float M[6]) {
    // srcPts: [x0, y0, x1, y1, x2, y2]
    // Solve: dst = M * src
    
    float a00 = srcPts[0], a01 = srcPts[1], a02 = 1;
    float a10 = srcPts[2], a11 = srcPts[3], a12 = 1;
    float a20 = srcPts[4], a21 = srcPts[5], a22 = 1;
    
    float det = a00 * (a11 * a22 - a21 * a12) -
                a01 * (a10 * a22 - a20 * a12) +
                a02 * (a10 * a21 - a20 * a11);
    
    if (std::abs(det) < 1e-10f) return false;
    
    float invDet = 1.0f / det;
    
    // Inverse of source matrix
    float inv[9];
    inv[0] = (a11 * a22 - a21 * a12) * invDet;
    inv[1] = (a02 * a21 - a01 * a22) * invDet;
    inv[2] = (a01 * a12 - a02 * a11) * invDet;
    inv[3] = (a12 * a20 - a10 * a22) * invDet;
    inv[4] = (a00 * a22 - a02 * a20) * invDet;
    inv[5] = (a02 * a10 - a00 * a12) * invDet;
    inv[6] = (a10 * a21 - a11 * a20) * invDet;
    inv[7] = (a01 * a20 - a00 * a21) * invDet;
    inv[8] = (a00 * a11 - a01 * a10) * invDet;
    
    // M = dst * inv(src)
    M[0] = dstPts[0] * inv[0] + dstPts[2] * inv[3] + dstPts[4] * inv[6];
    M[1] = dstPts[0] * inv[1] + dstPts[2] * inv[4] + dstPts[4] * inv[7];
    M[2] = dstPts[0] * inv[2] + dstPts[2] * inv[5] + dstPts[4] * inv[8];
    M[3] = dstPts[1] * inv[0] + dstPts[3] * inv[3] + dstPts[5] * inv[6];
    M[4] = dstPts[1] * inv[1] + dstPts[3] * inv[4] + dstPts[5] * inv[7];
    M[5] = dstPts[1] * inv[2] + dstPts[3] * inv[5] + dstPts[5] * inv[8];
    
    return true;
}

/**
 * @brief Invert a 2x3 affine matrix
 */
inline bool invertAffineTransform(const float M[6], float Minv[6]) {
    float det = M[0] * M[4] - M[1] * M[3];
    if (std::abs(det) < 1e-10f) return false;
    
    float invDet = 1.0f / det;
    
    Minv[0] = M[4] * invDet;
    Minv[1] = -M[1] * invDet;
    Minv[2] = (M[1] * M[5] - M[4] * M[2]) * invDet;
    Minv[3] = -M[3] * invDet;
    Minv[4] = M[0] * invDet;
    Minv[5] = (M[3] * M[2] - M[0] * M[5]) * invDet;
    
    return true;
}

// ============================================================================
// Sampling Functions
// ============================================================================

namespace detail {

inline float reflectIndex(float idx, int size) {
    if (size <= 1) return 0;
    int period = 2 * (size - 1);
    int i = static_cast<int>(idx);
    if (i < 0) i = period - ((-i) % period);
    else i = i % period;
    return (i < size) ? static_cast<float>(i) : static_cast<float>(period - i);
}

inline float wrapIndex(float idx, int size) {
    if (size <= 0) return 0;
    int i = static_cast<int>(idx) % size;
    if (i < 0) i += size;
    return static_cast<float>(i);
}

inline void mapCoords(float x, float y, int w, int h, BorderMode mode,
                     int& mx, int& my, bool& valid) {
    valid = true;
    
    if (mode == BorderMode::REPLICATE) {
        mx = std::clamp(static_cast<int>(x), 0, w - 1);
        my = std::clamp(static_cast<int>(y), 0, h - 1);
    } else if (mode == BorderMode::REFLECT) {
        mx = static_cast<int>(reflectIndex(x, w));
        my = static_cast<int>(reflectIndex(y, h));
    } else if (mode == BorderMode::WRAP) {
        mx = static_cast<int>(wrapIndex(x, w));
        my = static_cast<int>(wrapIndex(y, h));
    } else {
        // CONSTANT
        if (x < 0 || x >= w || y < 0 || y >= h) {
            valid = false;
            mx = 0;
            my = 0;
        } else {
            mx = static_cast<int>(x);
            my = static_cast<int>(y);
        }
    }
}

inline float sampleNearest(const uint8_t* img, int w, int h, int c,
                          float x, float y, int channel,
                          BorderMode mode, float constantValue) {
    int mx, my;
    bool valid;
    mapCoords(std::round(x), std::round(y), w, h, mode, mx, my, valid);
    
    if (!valid) return constantValue;
    return static_cast<float>(img[(my * w + mx) * c + channel]);
}

inline float sampleBilinear(const uint8_t* img, int w, int h, int c,
                           float x, float y, int channel,
                           BorderMode mode, float constantValue) {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    float fx = x - x0;
    float fy = y - y0;
    
    int mx[4], my[4];
    bool valid[4];
    
    mapCoords(static_cast<float>(x0), static_cast<float>(y0), w, h, mode, mx[0], my[0], valid[0]);
    mapCoords(static_cast<float>(x0 + 1), static_cast<float>(y0), w, h, mode, mx[1], my[1], valid[1]);
    mapCoords(static_cast<float>(x0), static_cast<float>(y0 + 1), w, h, mode, mx[2], my[2], valid[2]);
    mapCoords(static_cast<float>(x0 + 1), static_cast<float>(y0 + 1), w, h, mode, mx[3], my[3], valid[3]);
    
    float v00 = valid[0] ? static_cast<float>(img[(my[0] * w + mx[0]) * c + channel]) : constantValue;
    float v10 = valid[1] ? static_cast<float>(img[(my[1] * w + mx[1]) * c + channel]) : constantValue;
    float v01 = valid[2] ? static_cast<float>(img[(my[2] * w + mx[2]) * c + channel]) : constantValue;
    float v11 = valid[3] ? static_cast<float>(img[(my[3] * w + mx[3]) * c + channel]) : constantValue;
    
    return v00 * (1 - fx) * (1 - fy) +
           v10 * fx * (1 - fy) +
           v01 * (1 - fx) * fy +
           v11 * fx * fy;
}

} // namespace detail

// ============================================================================
// Resize Functions
// ============================================================================

/**
 * @brief Resize image using nearest neighbor interpolation
 */
inline void resizeNearest(const uint8_t* src, int srcW, int srcH, int channels,
                         uint8_t* dst, int dstW, int dstH) {
    float scaleX = static_cast<float>(srcW) / dstW;
    float scaleY = static_cast<float>(srcH) / dstH;
    
    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            float srcX = (x + 0.5f) * scaleX - 0.5f;
            float srcY = (y + 0.5f) * scaleY - 0.5f;
            
            int sx = std::clamp(static_cast<int>(std::round(srcX)), 0, srcW - 1);
            int sy = std::clamp(static_cast<int>(std::round(srcY)), 0, srcH - 1);
            
            for (int c = 0; c < channels; ++c) {
                dst[(y * dstW + x) * channels + c] = src[(sy * srcW + sx) * channels + c];
            }
        }
    }
}

/**
 * @brief Resize image using bilinear interpolation
 */
inline void resizeBilinear(const uint8_t* src, int srcW, int srcH, int channels,
                          uint8_t* dst, int dstW, int dstH) {
    float scaleX = static_cast<float>(srcW) / dstW;
    float scaleY = static_cast<float>(srcH) / dstH;
    
    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            float srcX = (x + 0.5f) * scaleX - 0.5f;
            float srcY = (y + 0.5f) * scaleY - 0.5f;
            
            int x0 = static_cast<int>(std::floor(srcX));
            int y0 = static_cast<int>(std::floor(srcY));
            float fx = srcX - x0;
            float fy = srcY - y0;
            
            int x0c = std::clamp(x0, 0, srcW - 1);
            int x1c = std::clamp(x0 + 1, 0, srcW - 1);
            int y0c = std::clamp(y0, 0, srcH - 1);
            int y1c = std::clamp(y0 + 1, 0, srcH - 1);
            
            for (int c = 0; c < channels; ++c) {
                float v00 = src[(y0c * srcW + x0c) * channels + c];
                float v10 = src[(y0c * srcW + x1c) * channels + c];
                float v01 = src[(y1c * srcW + x0c) * channels + c];
                float v11 = src[(y1c * srcW + x1c) * channels + c];
                
                float val = v00 * (1 - fx) * (1 - fy) +
                           v10 * fx * (1 - fy) +
                           v01 * (1 - fx) * fy +
                           v11 * fx * fy;
                
                dst[(y * dstW + x) * channels + c] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
            }
        }
    }
}

/**
 * @brief Resize image
 */
inline void resize(const uint8_t* src, int srcW, int srcH, int channels,
                  uint8_t* dst, int dstW, int dstH,
                  InterpolationMode interp = InterpolationMode::LINEAR) {
    if (interp == InterpolationMode::NEAREST) {
        resizeNearest(src, srcW, srcH, channels, dst, dstW, dstH);
    } else {
        resizeBilinear(src, srcW, srcH, channels, dst, dstW, dstH);
    }
}

// ============================================================================
// Warp Functions
// ============================================================================

/**
 * @brief Apply affine transformation
 */
inline void warpAffine(const uint8_t* src, int srcW, int srcH, int channels,
                      uint8_t* dst, int dstW, int dstH,
                      const float M[6],
                      InterpolationMode interp = InterpolationMode::LINEAR,
                      BorderMode borderMode = BorderMode::CONSTANT,
                      float constantValue = 0) {
    
    // Compute inverse transformation
    float Minv[6];
    if (!invertAffineTransform(M, Minv)) {
        std::fill(dst, dst + dstW * dstH * channels, static_cast<uint8_t>(constantValue));
        return;
    }
    
    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            float srcX = Minv[0] * x + Minv[1] * y + Minv[2];
            float srcY = Minv[3] * x + Minv[4] * y + Minv[5];
            
            for (int c = 0; c < channels; ++c) {
                float val;
                if (interp == InterpolationMode::NEAREST) {
                    val = detail::sampleNearest(src, srcW, srcH, channels, srcX, srcY, c, borderMode, constantValue);
                } else {
                    val = detail::sampleBilinear(src, srcW, srcH, channels, srcX, srcY, c, borderMode, constantValue);
                }
                dst[(y * dstW + x) * channels + c] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
            }
        }
    }
}

/**
 * @brief Apply perspective transformation
 */
inline void warpPerspective(const uint8_t* src, int srcW, int srcH, int channels,
                           uint8_t* dst, int dstW, int dstH,
                           const float H[9],
                           InterpolationMode interp = InterpolationMode::LINEAR,
                           BorderMode borderMode = BorderMode::CONSTANT,
                           float constantValue = 0) {
    
    // Compute inverse homography
    float det = H[0] * (H[4] * H[8] - H[5] * H[7]) -
                H[1] * (H[3] * H[8] - H[5] * H[6]) +
                H[2] * (H[3] * H[7] - H[4] * H[6]);
    
    if (std::abs(det) < 1e-10f) {
        std::fill(dst, dst + dstW * dstH * channels, static_cast<uint8_t>(constantValue));
        return;
    }
    
    float Hinv[9];
    float invDet = 1.0f / det;
    Hinv[0] = (H[4] * H[8] - H[5] * H[7]) * invDet;
    Hinv[1] = (H[2] * H[7] - H[1] * H[8]) * invDet;
    Hinv[2] = (H[1] * H[5] - H[2] * H[4]) * invDet;
    Hinv[3] = (H[5] * H[6] - H[3] * H[8]) * invDet;
    Hinv[4] = (H[0] * H[8] - H[2] * H[6]) * invDet;
    Hinv[5] = (H[2] * H[3] - H[0] * H[5]) * invDet;
    Hinv[6] = (H[3] * H[7] - H[4] * H[6]) * invDet;
    Hinv[7] = (H[1] * H[6] - H[0] * H[7]) * invDet;
    Hinv[8] = (H[0] * H[4] - H[1] * H[3]) * invDet;
    
    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            float w = Hinv[6] * x + Hinv[7] * y + Hinv[8];
            if (std::abs(w) < 1e-10f) {
                for (int c = 0; c < channels; ++c) {
                    dst[(y * dstW + x) * channels + c] = static_cast<uint8_t>(constantValue);
                }
                continue;
            }
            
            float srcX = (Hinv[0] * x + Hinv[1] * y + Hinv[2]) / w;
            float srcY = (Hinv[3] * x + Hinv[4] * y + Hinv[5]) / w;
            
            for (int c = 0; c < channels; ++c) {
                float val;
                if (interp == InterpolationMode::NEAREST) {
                    val = detail::sampleNearest(src, srcW, srcH, channels, srcX, srcY, c, borderMode, constantValue);
                } else {
                    val = detail::sampleBilinear(src, srcW, srcH, channels, srcX, srcY, c, borderMode, constantValue);
                }
                dst[(y * dstW + x) * channels + c] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
            }
        }
    }
}

// ============================================================================
// Rotation Functions
// ============================================================================

/**
 * @brief Rotate image
 */
inline void rotate(const uint8_t* src, int srcW, int srcH, int channels,
                  uint8_t* dst, int dstW, int dstH,
                  float angleDegrees, float centerX, float centerY, float scale = 1.0f,
                  InterpolationMode interp = InterpolationMode::LINEAR,
                  BorderMode borderMode = BorderMode::CONSTANT,
                  float constantValue = 0) {
    
    float M[6];
    getRotationMatrix2D(centerX, centerY, angleDegrees, scale, M);
    warpAffine(src, srcW, srcH, channels, dst, dstW, dstH, M, interp, borderMode, constantValue);
}

/**
 * @brief Rotate image with automatic canvas expansion
 */
inline void rotateExpand(const uint8_t* src, int srcW, int srcH, int channels,
                        std::vector<uint8_t>& dst, int& dstW, int& dstH,
                        float angleDegrees,
                        InterpolationMode interp = InterpolationMode::LINEAR,
                        float constantValue = 0) {
    
    float centerX = srcW * 0.5f;
    float centerY = srcH * 0.5f;
    
    float M[6];
    getRotationMatrix2D(centerX, centerY, angleDegrees, 1.0f, M);
    
    // Compute bounds by rotating corners
    float corners[4][2] = {
        {0, 0}, 
        {static_cast<float>(srcW - 1), 0},
        {0, static_cast<float>(srcH - 1)},
        {static_cast<float>(srcW - 1), static_cast<float>(srcH - 1)}
    };
    
    float minX = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::lowest();
    
    for (auto& c : corners) {
        float x = M[0] * c[0] + M[1] * c[1] + M[2];
        float y = M[3] * c[0] + M[4] * c[1] + M[5];
        minX = std::min(minX, x);
        maxX = std::max(maxX, x);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
    }
    
    dstW = static_cast<int>(std::ceil(maxX - minX + 1));
    dstH = static_cast<int>(std::ceil(maxY - minY + 1));
    
    // Adjust transformation to translate output
    float M2[6];
    M2[0] = M[0];
    M2[1] = M[1];
    M2[2] = M[2] - minX;
    M2[3] = M[3];
    M2[4] = M[4];
    M2[5] = M[5] - minY;
    
    dst.resize(dstW * dstH * channels);
    warpAffine(src, srcW, srcH, channels, dst.data(), dstW, dstH, M2, interp, BorderMode::CONSTANT, constantValue);
}

/**
 * @brief Rotate 90 degrees (fast path)
 */
inline void rotate90(const uint8_t* src, int srcW, int srcH, int channels,
                    uint8_t* dst) {
    // Output is srcH x srcW
    for (int y = 0; y < srcW; ++y) {
        for (int x = 0; x < srcH; ++x) {
            int srcX = srcW - 1 - y;
            int srcY = x;
            for (int c = 0; c < channels; ++c) {
                dst[(y * srcH + x) * channels + c] = src[(srcY * srcW + srcX) * channels + c];
            }
        }
    }
}

/**
 * @brief Rotate 180 degrees (fast path)
 */
inline void rotate180(const uint8_t* src, int srcW, int srcH, int channels,
                     uint8_t* dst) {
    for (int y = 0; y < srcH; ++y) {
        for (int x = 0; x < srcW; ++x) {
            int srcX = srcW - 1 - x;
            int srcY = srcH - 1 - y;
            for (int c = 0; c < channels; ++c) {
                dst[(y * srcW + x) * channels + c] = src[(srcY * srcW + srcX) * channels + c];
            }
        }
    }
}

/**
 * @brief Rotate 270 degrees (fast path)
 */
inline void rotate270(const uint8_t* src, int srcW, int srcH, int channels,
                     uint8_t* dst) {
    // Output is srcH x srcW
    for (int y = 0; y < srcW; ++y) {
        for (int x = 0; x < srcH; ++x) {
            int srcX = y;
            int srcY = srcH - 1 - x;
            for (int c = 0; c < channels; ++c) {
                dst[(y * srcH + x) * channels + c] = src[(srcY * srcW + srcX) * channels + c];
            }
        }
    }
}

// ============================================================================
// Flip Functions
// ============================================================================

/**
 * @brief Flip image horizontally
 */
inline void flipHorizontal(const uint8_t* src, int w, int h, int channels, uint8_t* dst) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int srcX = w - 1 - x;
            for (int c = 0; c < channels; ++c) {
                dst[(y * w + x) * channels + c] = src[(y * w + srcX) * channels + c];
            }
        }
    }
}

/**
 * @brief Flip image vertically
 */
inline void flipVertical(const uint8_t* src, int w, int h, int channels, uint8_t* dst) {
    for (int y = 0; y < h; ++y) {
        int srcY = h - 1 - y;
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) {
                dst[(y * w + x) * channels + c] = src[(srcY * w + x) * channels + c];
            }
        }
    }
}

/**
 * @brief Flip image (both axes)
 */
inline void flip(const uint8_t* src, int w, int h, int channels, uint8_t* dst, int flipCode) {
    if (flipCode == 0) {
        flipVertical(src, w, h, channels, dst);
    } else if (flipCode > 0) {
        flipHorizontal(src, w, h, channels, dst);
    } else {
        // Both
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int srcX = w - 1 - x;
                int srcY = h - 1 - y;
                for (int c = 0; c < channels; ++c) {
                    dst[(y * w + x) * channels + c] = src[(srcY * w + srcX) * channels + c];
                }
            }
        }
    }
}

// ============================================================================
// Remap
// ============================================================================

/**
 * @brief Remap using floating point maps
 */
inline void remap(const uint8_t* src, int srcW, int srcH, int channels,
                 uint8_t* dst, int dstW, int dstH,
                 const float* mapX, const float* mapY,
                 InterpolationMode interp = InterpolationMode::LINEAR,
                 BorderMode borderMode = BorderMode::CONSTANT,
                 float constantValue = 0) {
    
    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            int idx = y * dstW + x;
            float srcX = mapX[idx];
            float srcY = mapY[idx];
            
            for (int c = 0; c < channels; ++c) {
                float val;
                if (interp == InterpolationMode::NEAREST) {
                    val = detail::sampleNearest(src, srcW, srcH, channels, srcX, srcY, c, borderMode, constantValue);
                } else {
                    val = detail::sampleBilinear(src, srcW, srcH, channels, srcX, srcY, c, borderMode, constantValue);
                }
                dst[idx * channels + c] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
            }
        }
    }
}

// ============================================================================
// Crop and Pad
// ============================================================================

/**
 * @brief Crop region from image
 */
inline void crop(const uint8_t* src, int srcW, int srcH, int channels,
                uint8_t* dst, int x, int y, int cropW, int cropH) {
    for (int dy = 0; dy < cropH; ++dy) {
        for (int dx = 0; dx < cropW; ++dx) {
            int sx = x + dx;
            int sy = y + dy;
            if (sx >= 0 && sx < srcW && sy >= 0 && sy < srcH) {
                for (int c = 0; c < channels; ++c) {
                    dst[(dy * cropW + dx) * channels + c] = src[(sy * srcW + sx) * channels + c];
                }
            } else {
                for (int c = 0; c < channels; ++c) {
                    dst[(dy * cropW + dx) * channels + c] = 0;
                }
            }
        }
    }
}

/**
 * @brief Pad image with border
 */
inline void copyMakeBorder(const uint8_t* src, int srcW, int srcH, int channels,
                          uint8_t* dst,
                          int top, int bottom, int left, int right,
                          BorderMode borderMode = BorderMode::CONSTANT,
                          uint8_t value = 0) {
    int dstW = srcW + left + right;
    int dstH = srcH + top + bottom;
    
    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            int srcX = x - left;
            int srcY = y - top;
            
            bool inBounds = (srcX >= 0 && srcX < srcW && srcY >= 0 && srcY < srcH);
            
            if (inBounds) {
                for (int c = 0; c < channels; ++c) {
                    dst[(y * dstW + x) * channels + c] = src[(srcY * srcW + srcX) * channels + c];
                }
            } else if (borderMode == BorderMode::CONSTANT) {
                for (int c = 0; c < channels; ++c) {
                    dst[(y * dstW + x) * channels + c] = value;
                }
            } else if (borderMode == BorderMode::REPLICATE) {
                int mx = std::clamp(srcX, 0, srcW - 1);
                int my = std::clamp(srcY, 0, srcH - 1);
                for (int c = 0; c < channels; ++c) {
                    dst[(y * dstW + x) * channels + c] = src[(my * srcW + mx) * channels + c];
                }
            } else if (borderMode == BorderMode::REFLECT) {
                int mx = static_cast<int>(detail::reflectIndex(static_cast<float>(srcX), srcW));
                int my = static_cast<int>(detail::reflectIndex(static_cast<float>(srcY), srcH));
                for (int c = 0; c < channels; ++c) {
                    dst[(y * dstW + x) * channels + c] = src[(my * srcW + mx) * channels + c];
                }
            } else { // WRAP
                int mx = static_cast<int>(detail::wrapIndex(static_cast<float>(srcX), srcW));
                int my = static_cast<int>(detail::wrapIndex(static_cast<float>(srcY), srcH));
                for (int c = 0; c < channels; ++c) {
                    dst[(y * dstW + x) * channels + c] = src[(my * srcW + mx) * channels + c];
                }
            }
        }
    }
}

} // namespace transform
} // namespace neurova

#endif // NEUROVA_TRANSFORM_HPP
