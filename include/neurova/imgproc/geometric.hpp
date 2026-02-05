// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file geometric.hpp
 * @brief Geometric image transformations
 */

#ifndef NEUROVA_IMGPROC_GEOMETRIC_HPP
#define NEUROVA_IMGPROC_GEOMETRIC_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <array>

namespace neurova {
namespace imgproc {

// Interpolation flags
constexpr int INTER_NEAREST = 0;
constexpr int INTER_LINEAR = 1;
constexpr int INTER_CUBIC = 2;
constexpr int INTER_AREA = 3;
constexpr int INTER_LANCZOS4 = 4;

// Border modes for remapping
constexpr int BORDER_CONSTANT = 0;
constexpr int BORDER_REPLICATE = 1;
constexpr int BORDER_REFLECT = 2;
constexpr int BORDER_WRAP = 3;
constexpr int BORDER_REFLECT101 = 4;

/**
 * @brief Bilinear interpolation at subpixel position
 */
inline float bilinearInterpolate(
    const float* image, int width, int height,
    float x, float y, int channels = 1, int channel = 0
) {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    x0 = std::clamp(x0, 0, width - 1);
    x1 = std::clamp(x1, 0, width - 1);
    y0 = std::clamp(y0, 0, height - 1);
    y1 = std::clamp(y1, 0, height - 1);
    
    float dx = x - std::floor(x);
    float dy = y - std::floor(y);
    
    float v00 = image[(y0 * width + x0) * channels + channel];
    float v01 = image[(y0 * width + x1) * channels + channel];
    float v10 = image[(y1 * width + x0) * channels + channel];
    float v11 = image[(y1 * width + x1) * channels + channel];
    
    return (1 - dx) * (1 - dy) * v00 +
           dx * (1 - dy) * v01 +
           (1 - dx) * dy * v10 +
           dx * dy * v11;
}

/**
 * @brief Cubic interpolation coefficient
 */
inline float cubicWeight(float t, float a = -0.5f) {
    t = std::abs(t);
    if (t <= 1.0f) {
        return ((a + 2.0f) * t - (a + 3.0f)) * t * t + 1.0f;
    } else if (t < 2.0f) {
        return a * (((t - 5.0f) * t + 8.0f) * t - 4.0f);
    }
    return 0.0f;
}

/**
 * @brief Bicubic interpolation at subpixel position
 */
inline float bicubicInterpolate(
    const float* image, int width, int height,
    float x, float y, int channels = 1, int channel = 0
) {
    int x0 = static_cast<int>(std::floor(x)) - 1;
    int y0 = static_cast<int>(std::floor(y)) - 1;
    
    float sum = 0.0f;
    float weightSum = 0.0f;
    
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            int px = std::clamp(x0 + i, 0, width - 1);
            int py = std::clamp(y0 + j, 0, height - 1);
            
            float wx = cubicWeight(x - (x0 + i));
            float wy = cubicWeight(y - (y0 + j));
            float weight = wx * wy;
            
            sum += image[(py * width + px) * channels + channel] * weight;
            weightSum += weight;
        }
    }
    
    return (weightSum > 0) ? (sum / weightSum) : 0.0f;
}

/**
 * @brief Resize image
 */
inline std::vector<float> resize(
    const float* input, int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    int interpolation = INTER_LINEAR,
    int channels = 1
) {
    std::vector<float> output(dstWidth * dstHeight * channels);
    
    float scaleX = static_cast<float>(srcWidth) / dstWidth;
    float scaleY = static_cast<float>(srcHeight) / dstHeight;
    
    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            float srcX = (x + 0.5f) * scaleX - 0.5f;
            float srcY = (y + 0.5f) * scaleY - 0.5f;
            
            for (int c = 0; c < channels; ++c) {
                float value;
                
                if (interpolation == INTER_NEAREST) {
                    int sx = static_cast<int>(srcX + 0.5f);
                    int sy = static_cast<int>(srcY + 0.5f);
                    sx = std::clamp(sx, 0, srcWidth - 1);
                    sy = std::clamp(sy, 0, srcHeight - 1);
                    value = input[(sy * srcWidth + sx) * channels + c];
                } else if (interpolation == INTER_LINEAR) {
                    value = bilinearInterpolate(input, srcWidth, srcHeight,
                                               srcX, srcY, channels, c);
                } else if (interpolation == INTER_CUBIC) {
                    value = bicubicInterpolate(input, srcWidth, srcHeight,
                                              srcX, srcY, channels, c);
                } else {
                    value = bilinearInterpolate(input, srcWidth, srcHeight,
                                               srcX, srcY, channels, c);
                }
                
                output[(y * dstWidth + x) * channels + c] = value;
            }
        }
    }
    
    return output;
}

/**
 * @brief Get 2x3 rotation matrix
 */
inline std::array<float, 6> getRotationMatrix2D(
    float centerX, float centerY,
    float angle,  // in degrees
    float scale = 1.0f
) {
    float radians = angle * 3.14159265f / 180.0f;
    float alpha = scale * std::cos(radians);
    float beta = scale * std::sin(radians);
    
    return {
        alpha, beta, (1 - alpha) * centerX - beta * centerY,
        -beta, alpha, beta * centerX + (1 - alpha) * centerY
    };
}

/**
 * @brief Apply affine transformation
 */
inline std::vector<float> warpAffine(
    const float* input, int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    const float* M,  // 2x3 transformation matrix
    int interpolation = INTER_LINEAR,
    int borderMode = BORDER_CONSTANT,
    float borderValue = 0.0f,
    int channels = 1
) {
    std::vector<float> output(dstWidth * dstHeight * channels, borderValue);
    
    // Compute inverse transformation for backward mapping
    float det = M[0] * M[4] - M[1] * M[3];
    if (std::abs(det) < 1e-10f) return output;
    
    float invDet = 1.0f / det;
    float invM[6] = {
        M[4] * invDet, -M[1] * invDet, (M[1] * M[5] - M[2] * M[4]) * invDet,
        -M[3] * invDet, M[0] * invDet, (M[2] * M[3] - M[0] * M[5]) * invDet
    };
    
    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            float srcX = invM[0] * x + invM[1] * y + invM[2];
            float srcY = invM[3] * x + invM[4] * y + invM[5];
            
            if (srcX < 0 || srcX >= srcWidth || srcY < 0 || srcY >= srcHeight) {
                continue;  // Use border value
            }
            
            for (int c = 0; c < channels; ++c) {
                float value;
                if (interpolation == INTER_NEAREST) {
                    int sx = static_cast<int>(srcX + 0.5f);
                    int sy = static_cast<int>(srcY + 0.5f);
                    sx = std::clamp(sx, 0, srcWidth - 1);
                    sy = std::clamp(sy, 0, srcHeight - 1);
                    value = input[(sy * srcWidth + sx) * channels + c];
                } else {
                    value = bilinearInterpolate(input, srcWidth, srcHeight,
                                               srcX, srcY, channels, c);
                }
                output[(y * dstWidth + x) * channels + c] = value;
            }
        }
    }
    
    return output;
}

/**
 * @brief Get perspective transformation matrix
 */
inline std::array<float, 9> getPerspectiveTransform(
    const float srcPoints[4][2],
    const float dstPoints[4][2]
) {
    // Solve for perspective transformation using DLT (Direct Linear Transform)
    // H * src = dst (homogeneous coordinates)
    
    // Build matrix A for Ah = 0
    float A[8][9];
    for (int i = 0; i < 4; ++i) {
        float x = srcPoints[i][0];
        float y = srcPoints[i][1];
        float u = dstPoints[i][0];
        float v = dstPoints[i][1];
        
        A[2*i][0] = -x; A[2*i][1] = -y; A[2*i][2] = -1;
        A[2*i][3] = 0;  A[2*i][4] = 0;  A[2*i][5] = 0;
        A[2*i][6] = u*x; A[2*i][7] = u*y; A[2*i][8] = u;
        
        A[2*i+1][0] = 0;  A[2*i+1][1] = 0;  A[2*i+1][2] = 0;
        A[2*i+1][3] = -x; A[2*i+1][4] = -y; A[2*i+1][5] = -1;
        A[2*i+1][6] = v*x; A[2*i+1][7] = v*y; A[2*i+1][8] = v;
    }
    
    // Solve using simple LU decomposition (simplified)
    // For robustness, use proper SVD in production code
    std::array<float, 9> H = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    
    // Simple solution by setting h33 = 1 and solving linear system
    float b[8];
    float Amod[8][8];
    
    for (int i = 0; i < 8; ++i) {
        b[i] = -A[i][8];
        for (int j = 0; j < 8; ++j) {
            Amod[i][j] = A[i][j];
        }
    }
    
    // Gaussian elimination
    for (int col = 0; col < 8; ++col) {
        // Find pivot
        int maxRow = col;
        for (int row = col + 1; row < 8; ++row) {
            if (std::abs(Amod[row][col]) > std::abs(Amod[maxRow][col])) {
                maxRow = row;
            }
        }
        
        // Swap rows
        for (int j = 0; j < 8; ++j) {
            std::swap(Amod[col][j], Amod[maxRow][j]);
        }
        std::swap(b[col], b[maxRow]);
        
        // Eliminate
        for (int row = col + 1; row < 8; ++row) {
            if (std::abs(Amod[col][col]) < 1e-10f) continue;
            float factor = Amod[row][col] / Amod[col][col];
            for (int j = col; j < 8; ++j) {
                Amod[row][j] -= factor * Amod[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    
    // Back substitution
    float h[8];
    for (int i = 7; i >= 0; --i) {
        h[i] = b[i];
        for (int j = i + 1; j < 8; ++j) {
            h[i] -= Amod[i][j] * h[j];
        }
        if (std::abs(Amod[i][i]) > 1e-10f) {
            h[i] /= Amod[i][i];
        }
    }
    
    H = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0f};
    
    return H;
}

/**
 * @brief Apply perspective transformation
 */
inline std::vector<float> warpPerspective(
    const float* input, int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    const float* H,  // 3x3 transformation matrix
    int interpolation = INTER_LINEAR,
    int borderMode = BORDER_CONSTANT,
    float borderValue = 0.0f,
    int channels = 1
) {
    std::vector<float> output(dstWidth * dstHeight * channels, borderValue);
    
    // Compute inverse transformation
    float det = H[0] * (H[4] * H[8] - H[5] * H[7]) -
                H[1] * (H[3] * H[8] - H[5] * H[6]) +
                H[2] * (H[3] * H[7] - H[4] * H[6]);
    
    if (std::abs(det) < 1e-10f) return output;
    
    float invDet = 1.0f / det;
    float invH[9] = {
        (H[4] * H[8] - H[5] * H[7]) * invDet,
        (H[2] * H[7] - H[1] * H[8]) * invDet,
        (H[1] * H[5] - H[2] * H[4]) * invDet,
        (H[5] * H[6] - H[3] * H[8]) * invDet,
        (H[0] * H[8] - H[2] * H[6]) * invDet,
        (H[2] * H[3] - H[0] * H[5]) * invDet,
        (H[3] * H[7] - H[4] * H[6]) * invDet,
        (H[1] * H[6] - H[0] * H[7]) * invDet,
        (H[0] * H[4] - H[1] * H[3]) * invDet
    };
    
    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            float w = invH[6] * x + invH[7] * y + invH[8];
            if (std::abs(w) < 1e-10f) continue;
            
            float srcX = (invH[0] * x + invH[1] * y + invH[2]) / w;
            float srcY = (invH[3] * x + invH[4] * y + invH[5]) / w;
            
            if (srcX < 0 || srcX >= srcWidth || srcY < 0 || srcY >= srcHeight) {
                continue;
            }
            
            for (int c = 0; c < channels; ++c) {
                float value;
                if (interpolation == INTER_NEAREST) {
                    int sx = static_cast<int>(srcX + 0.5f);
                    int sy = static_cast<int>(srcY + 0.5f);
                    sx = std::clamp(sx, 0, srcWidth - 1);
                    sy = std::clamp(sy, 0, srcHeight - 1);
                    value = input[(sy * srcWidth + sx) * channels + c];
                } else {
                    value = bilinearInterpolate(input, srcWidth, srcHeight,
                                               srcX, srcY, channels, c);
                }
                output[(y * dstWidth + x) * channels + c] = value;
            }
        }
    }
    
    return output;
}

/**
 * @brief Rotate image
 */
inline std::vector<float> rotate(
    const float* input, int width, int height,
    float angle,
    float scale = 1.0f,
    int interpolation = INTER_LINEAR,
    int channels = 1
) {
    float centerX = width / 2.0f;
    float centerY = height / 2.0f;
    
    auto M = getRotationMatrix2D(centerX, centerY, angle, scale);
    
    return warpAffine(input, width, height, width, height,
                     M.data(), interpolation, BORDER_CONSTANT, 0.0f, channels);
}

/**
 * @brief Flip image horizontally
 */
inline std::vector<float> flipHorizontal(
    const float* input, int width, int height, int channels = 1
) {
    std::vector<float> output(width * height * channels);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int srcX = width - 1 - x;
            for (int c = 0; c < channels; ++c) {
                output[(y * width + x) * channels + c] = 
                    input[(y * width + srcX) * channels + c];
            }
        }
    }
    
    return output;
}

/**
 * @brief Flip image vertically
 */
inline std::vector<float> flipVertical(
    const float* input, int width, int height, int channels = 1
) {
    std::vector<float> output(width * height * channels);
    
    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                output[(y * width + x) * channels + c] = 
                    input[(srcY * width + x) * channels + c];
            }
        }
    }
    
    return output;
}

/**
 * @brief Flip image (standard-style)
 * @param flipCode 0=vertical, 1=horizontal, -1=both
 */
inline std::vector<float> flip(
    const float* input, int width, int height, int flipCode, int channels = 1
) {
    if (flipCode == 0) {
        return flipVertical(input, width, height, channels);
    } else if (flipCode > 0) {
        return flipHorizontal(input, width, height, channels);
    } else {
        auto temp = flipHorizontal(input, width, height, channels);
        return flipVertical(temp.data(), width, height, channels);
    }
}

/**
 * @brief Transpose image
 */
inline std::vector<float> transpose(
    const float* input, int width, int height, int channels = 1
) {
    std::vector<float> output(width * height * channels);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                output[(x * height + y) * channels + c] = 
                    input[(y * width + x) * channels + c];
            }
        }
    }
    
    return output;
}

/**
 * @brief Crop region from image
 */
inline std::vector<float> crop(
    const float* input, int width, int height,
    int x, int y, int cropWidth, int cropHeight,
    int channels = 1
) {
    std::vector<float> output(cropWidth * cropHeight * channels);
    
    for (int cy = 0; cy < cropHeight; ++cy) {
        for (int cx = 0; cx < cropWidth; ++cx) {
            int srcX = std::clamp(x + cx, 0, width - 1);
            int srcY = std::clamp(y + cy, 0, height - 1);
            
            for (int c = 0; c < channels; ++c) {
                output[(cy * cropWidth + cx) * channels + c] = 
                    input[(srcY * width + srcX) * channels + c];
            }
        }
    }
    
    return output;
}

/**
 * @brief Generic geometric remapping
 */
inline std::vector<float> remap(
    const float* input, int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    const float* mapX, const float* mapY,
    int interpolation = INTER_LINEAR,
    int borderMode = BORDER_CONSTANT,
    float borderValue = 0.0f,
    int channels = 1
) {
    std::vector<float> output(dstWidth * dstHeight * channels, borderValue);
    
    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            float srcX = mapX[y * dstWidth + x];
            float srcY = mapY[y * dstWidth + x];
            
            if (srcX < 0 || srcX >= srcWidth || srcY < 0 || srcY >= srcHeight) {
                continue;
            }
            
            for (int c = 0; c < channels; ++c) {
                float value;
                if (interpolation == INTER_NEAREST) {
                    int sx = static_cast<int>(srcX + 0.5f);
                    int sy = static_cast<int>(srcY + 0.5f);
                    sx = std::clamp(sx, 0, srcWidth - 1);
                    sy = std::clamp(sy, 0, srcHeight - 1);
                    value = input[(sy * srcWidth + sx) * channels + c];
                } else {
                    value = bilinearInterpolate(input, srcWidth, srcHeight,
                                               srcX, srcY, channels, c);
                }
                output[(y * dstWidth + x) * channels + c] = value;
            }
        }
    }
    
    return output;
}

/**
 * @brief Get affine transformation from 3 point pairs
 */
inline std::array<float, 6> getAffineTransform(
    const float srcPoints[3][2],
    const float dstPoints[3][2]
) {
    // Solve [dst] = [M] * [src]
    // [ u0 ]   [ a b c ] [ x0 ]
    // [ v0 ] = [ d e f ] [ y0 ]
    // [ u1 ]             [ x1 ]
    // [ v1 ]             [ y1 ]
    // [ u2 ]             [ x2 ]
    // [ v2 ]             [ y2 ]
    
    float x0 = srcPoints[0][0], y0 = srcPoints[0][1];
    float x1 = srcPoints[1][0], y1 = srcPoints[1][1];
    float x2 = srcPoints[2][0], y2 = srcPoints[2][1];
    
    float u0 = dstPoints[0][0], v0 = dstPoints[0][1];
    float u1 = dstPoints[1][0], v1 = dstPoints[1][1];
    float u2 = dstPoints[2][0], v2 = dstPoints[2][1];
    
    float det = x0 * (y1 - y2) - x1 * (y0 - y2) + x2 * (y0 - y1);
    if (std::abs(det) < 1e-10f) {
        return {1, 0, 0, 0, 1, 0};
    }
    
    float invDet = 1.0f / det;
    
    float a = ((u0 - u2) * (y1 - y2) - (u1 - u2) * (y0 - y2)) * invDet;
    float b = ((u1 - u2) * (x0 - x2) - (u0 - u2) * (x1 - x2)) * invDet;
    float c = u0 - a * x0 - b * y0;
    
    float d = ((v0 - v2) * (y1 - y2) - (v1 - v2) * (y0 - y2)) * invDet;
    float e = ((v1 - v2) * (x0 - x2) - (v0 - v2) * (x1 - x2)) * invDet;
    float f = v0 - d * x0 - e * y0;
    
    return {a, b, c, d, e, f};
}

} // namespace imgproc
} // namespace neurova

#endif // NEUROVA_IMGPROC_GEOMETRIC_HPP
