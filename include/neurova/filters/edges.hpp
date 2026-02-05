// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file edges.hpp
 * @brief Edge detection filters
 */

#ifndef NEUROVA_FILTERS_EDGES_HPP
#define NEUROVA_FILTERS_EDGES_HPP

#include "kernels.hpp"
#include "convolution.hpp"
#include "blur.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>

namespace neurova {
namespace filters {

/**
 * @brief Sobel edge detection
 * @param input Input grayscale image
 * @param width Image width
 * @param height Image height
 * @param dx Order of derivative in X
 * @param dy Order of derivative in Y
 * @param ksize Kernel size (1, 3, 5, or 7)
 * @param scale Scale factor for computed values
 * @param delta Value added to results
 * @param borderType Border handling
 * @return Gradient image
 */
inline std::vector<float> sobel(
    const float* input, int width, int height,
    int dx = 1, int dy = 0,
    int ksize = 3,
    float scale = 1.0f,
    float delta = 0.0f,
    int borderType = BORDER_REFLECT_101
) {
    auto [gx, gy] = sobelKernels(ksize);
    
    std::vector<float> result(width * height, 0.0f);
    
    if (dx > 0) {
        auto gradX = filter2D(input, width, height,
                             gx.data(), ksize, ksize,
                             -1, -1, 0.0f, borderType);
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] += gradX[i];
        }
    }
    
    if (dy > 0) {
        auto gradY = filter2D(input, width, height,
                             gy.data(), ksize, ksize,
                             -1, -1, 0.0f, borderType);
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] += gradY[i];
        }
    }
    
    // Apply scale and delta
    for (auto& v : result) {
        v = v * scale + delta;
    }
    
    return result;
}

/**
 * @brief Compute both Sobel gradients
 */
inline std::pair<std::vector<float>, std::vector<float>> sobelXY(
    const float* input, int width, int height,
    int ksize = 3,
    int borderType = BORDER_REFLECT_101
) {
    auto [gx, gy] = sobelKernels(ksize);
    
    auto gradX = filter2D(input, width, height,
                         gx.data(), ksize, ksize,
                         -1, -1, 0.0f, borderType);
    auto gradY = filter2D(input, width, height,
                         gy.data(), ksize, ksize,
                         -1, -1, 0.0f, borderType);
    
    return {gradX, gradY};
}

/**
 * @brief Scharr edge detection (more accurate than Sobel for 3x3)
 */
inline std::vector<float> scharr(
    const float* input, int width, int height,
    int dx = 1, int dy = 0,
    float scale = 1.0f,
    float delta = 0.0f,
    int borderType = BORDER_REFLECT_101
) {
    auto [gx, gy] = scharrKernels();
    
    std::vector<float> result(width * height, 0.0f);
    
    if (dx > 0) {
        auto gradX = filter2D(input, width, height,
                             gx.data(), 3, 3,
                             -1, -1, 0.0f, borderType);
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] += gradX[i];
        }
    }
    
    if (dy > 0) {
        auto gradY = filter2D(input, width, height,
                             gy.data(), 3, 3,
                             -1, -1, 0.0f, borderType);
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] += gradY[i];
        }
    }
    
    for (auto& v : result) {
        v = v * scale + delta;
    }
    
    return result;
}

/**
 * @brief Laplacian edge detection
 */
inline std::vector<float> laplacian(
    const float* input, int width, int height,
    int ksize = 3,
    float scale = 1.0f,
    float delta = 0.0f,
    int borderType = BORDER_REFLECT_101
) {
    auto kernel = laplacianKernel(ksize);
    int kernelSize = (ksize == 1) ? 3 : ksize;
    
    auto result = filter2D(input, width, height,
                          kernel.data(), kernelSize, kernelSize,
                          -1, -1, delta, borderType);
    
    if (scale != 1.0f) {
        for (auto& v : result) {
            v *= scale;
        }
    }
    
    return result;
}

/**
 * @brief Compute gradient magnitude
 */
inline std::vector<float> gradientMagnitude(
    const float* gradX, const float* gradY,
    int width, int height
) {
    std::vector<float> magnitude(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        magnitude[i] = std::sqrt(gradX[i] * gradX[i] + gradY[i] * gradY[i]);
    }
    
    return magnitude;
}

/**
 * @brief Compute gradient direction
 */
inline std::vector<float> gradientDirection(
    const float* gradX, const float* gradY,
    int width, int height
) {
    std::vector<float> direction(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        direction[i] = std::atan2(gradY[i], gradX[i]);
    }
    
    return direction;
}

/**
 * @brief Non-maximum suppression for edge thinning
 */
inline std::vector<float> nonMaxSuppressionEdge(
    const float* magnitude,
    const float* direction,
    int width, int height
) {
    std::vector<float> result(width * height, 0.0f);
    
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            float angle = direction[idx];
            float mag = magnitude[idx];
            
            // Convert angle to 0-180 range
            if (angle < 0) angle += 3.14159265f;
            float degrees = angle * 180.0f / 3.14159265f;
            
            float neighbor1, neighbor2;
            
            // Check neighbors based on gradient direction
            if ((degrees >= 0 && degrees < 22.5f) || (degrees >= 157.5f && degrees <= 180.0f)) {
                // Horizontal
                neighbor1 = magnitude[y * width + (x - 1)];
                neighbor2 = magnitude[y * width + (x + 1)];
            } else if (degrees >= 22.5f && degrees < 67.5f) {
                // Diagonal /
                neighbor1 = magnitude[(y - 1) * width + (x + 1)];
                neighbor2 = magnitude[(y + 1) * width + (x - 1)];
            } else if (degrees >= 67.5f && degrees < 112.5f) {
                // Vertical
                neighbor1 = magnitude[(y - 1) * width + x];
                neighbor2 = magnitude[(y + 1) * width + x];
            } else {
                // Diagonal \
                neighbor1 = magnitude[(y - 1) * width + (x - 1)];
                neighbor2 = magnitude[(y + 1) * width + (x + 1)];
            }
            
            // Suppress if not local maximum
            if (mag >= neighbor1 && mag >= neighbor2) {
                result[idx] = mag;
            }
        }
    }
    
    return result;
}

/**
 * @brief Hysteresis thresholding for Canny edge detection
 */
inline std::vector<float> hysteresisThreshold(
    const float* edges,
    int width, int height,
    float lowThreshold,
    float highThreshold
) {
    std::vector<float> result(width * height, 0.0f);
    std::vector<bool> visited(width * height, false);
    
    // Mark strong edges
    std::queue<std::pair<int, int>> edgeQueue;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            if (edges[idx] >= highThreshold) {
                result[idx] = 255.0f;
                visited[idx] = true;
                edgeQueue.push({x, y});
            }
        }
    }
    
    // Trace connected weak edges
    const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    
    while (!edgeQueue.empty()) {
        auto [x, y] = edgeQueue.front();
        edgeQueue.pop();
        
        for (int i = 0; i < 8; ++i) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            
            int nidx = ny * width + nx;
            if (visited[nidx]) continue;
            
            if (edges[nidx] >= lowThreshold) {
                result[nidx] = 255.0f;
                visited[nidx] = true;
                edgeQueue.push({nx, ny});
            }
        }
    }
    
    return result;
}

/**
 * @brief Canny edge detection
 * @param input Input grayscale image
 * @param width Image width
 * @param height Image height
 * @param lowThreshold Low threshold for hysteresis
 * @param highThreshold High threshold for hysteresis
 * @param apertureSize Sobel aperture size
 * @param L2gradient Use L2 norm for gradient magnitude
 * @return Edge map
 */
inline std::vector<float> canny(
    const float* input, int width, int height,
    float lowThreshold,
    float highThreshold,
    int apertureSize = 3,
    bool L2gradient = false
) {
    // Step 1: Gaussian blur
    auto blurred = gaussianBlur(input, width, height, 5, 1.4f);
    
    // Step 2: Compute gradients
    auto [gradX, gradY] = sobelXY(blurred.data(), width, height, apertureSize);
    
    // Step 3: Compute magnitude and direction
    std::vector<float> magnitude(width * height);
    std::vector<float> direction(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        if (L2gradient) {
            magnitude[i] = std::sqrt(gradX[i] * gradX[i] + gradY[i] * gradY[i]);
        } else {
            magnitude[i] = std::abs(gradX[i]) + std::abs(gradY[i]);
        }
        direction[i] = std::atan2(gradY[i], gradX[i]);
    }
    
    // Step 4: Non-maximum suppression
    auto suppressed = nonMaxSuppressionEdge(magnitude.data(), direction.data(),
                                           width, height);
    
    // Step 5: Hysteresis thresholding
    return hysteresisThreshold(suppressed.data(), width, height,
                              lowThreshold, highThreshold);
}

/**
 * @brief Simple edge detection using kernel
 */
inline std::vector<float> detectEdges(
    const float* input, int width, int height
) {
    auto kernel = edgeKernel();
    auto result = filter2D(input, width, height,
                          kernel.data(), 3, 3);
    
    // Take absolute value and clamp
    for (auto& v : result) {
        v = std::clamp(std::abs(v), 0.0f, 255.0f);
    }
    
    return result;
}

/**
 * @brief Prewitt edge detection
 */
inline std::pair<std::vector<float>, std::vector<float>> prewitt(
    const float* input, int width, int height,
    int borderType = BORDER_REFLECT_101
) {
    auto [gx, gy] = prewittKernels();
    
    auto gradX = filter2D(input, width, height,
                         gx.data(), 3, 3,
                         -1, -1, 0.0f, borderType);
    auto gradY = filter2D(input, width, height,
                         gy.data(), 3, 3,
                         -1, -1, 0.0f, borderType);
    
    return {gradX, gradY};
}

/**
 * @brief Roberts cross edge detection
 */
inline std::pair<std::vector<float>, std::vector<float>> roberts(
    const float* input, int width, int height
) {
    auto [gx, gy] = robertsKernels();
    
    std::vector<float> gradX(width * height);
    std::vector<float> gradY(width * height);
    
    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            int idx = y * width + x;
            float p00 = input[idx];
            float p01 = input[idx + 1];
            float p10 = input[(y + 1) * width + x];
            float p11 = input[(y + 1) * width + x + 1];
            
            gradX[idx] = p00 - p11;
            gradY[idx] = p01 - p10;
        }
    }
    
    return {gradX, gradY};
}

/**
 * @brief LoG (Laplacian of Gaussian) edge detection
 */
inline std::vector<float> logEdges(
    const float* input, int width, int height,
    int ksize = 5,
    float sigma = 1.0f
) {
    auto kernel = logKernel(ksize, sigma);
    return filter2D(input, width, height,
                   kernel.data(), ksize, ksize);
}

/**
 * @brief DoG (Difference of Gaussians) edge detection
 */
inline std::vector<float> dogEdges(
    const float* input, int width, int height,
    float sigma1 = 1.0f,
    float sigma2 = 2.0f
) {
    auto blurred1 = gaussianBlur(input, width, height, 0, sigma1);
    auto blurred2 = gaussianBlur(input, width, height, 0, sigma2);
    
    std::vector<float> result(width * height);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = blurred1[i] - blurred2[i];
    }
    
    return result;
}

/**
 * @brief Zero crossing detection for LoG/DoG edges
 */
inline std::vector<float> zeroCrossing(
    const float* input, int width, int height,
    float threshold = 0.0f
) {
    std::vector<float> result(width * height, 0.0f);
    
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            float center = input[idx];
            
            // Check horizontal and vertical neighbors
            bool crossing = false;
            
            float neighbors[] = {
                input[idx - 1], input[idx + 1],
                input[(y - 1) * width + x], input[(y + 1) * width + x]
            };
            
            for (float neighbor : neighbors) {
                if ((center > threshold && neighbor < -threshold) ||
                    (center < -threshold && neighbor > threshold)) {
                    crossing = true;
                    break;
                }
            }
            
            if (crossing) {
                result[idx] = 255.0f;
            }
        }
    }
    
    return result;
}

} // namespace filters
} // namespace neurova

#endif // NEUROVA_FILTERS_EDGES_HPP
