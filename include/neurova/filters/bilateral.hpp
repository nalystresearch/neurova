// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file bilateral.hpp
 * @brief Bilateral filtering for edge-preserving smoothing
 */

#ifndef NEUROVA_FILTERS_BILATERAL_HPP
#define NEUROVA_FILTERS_BILATERAL_HPP

#include "kernels.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace neurova {
namespace filters {

/**
 * @brief Bilateral filter - edge-preserving smoothing
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @param d Diameter of pixel neighborhood (use -1 for auto from sigmaSpace)
 * @param sigmaColor Filter sigma in color space
 * @param sigmaSpace Filter sigma in coordinate space
 * @param borderType Border handling type
 * @return Filtered image
 */
inline std::vector<float> bilateralFilter(
    const float* input, int width, int height,
    int d = -1,
    float sigmaColor = 75.0f,
    float sigmaSpace = 75.0f,
    int borderType = BORDER_REFLECT_101
) {
    // Determine filter size
    if (d <= 0) {
        d = static_cast<int>(std::ceil(sigmaSpace * 3) * 2 + 1);
    }
    if (d % 2 == 0) d += 1;
    
    int radius = d / 2;
    
    // Precompute spatial weights
    std::vector<float> spatialWeight(d * d);
    float spaceFactor = -0.5f / (sigmaSpace * sigmaSpace);
    
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            float dist2 = static_cast<float>(dx * dx + dy * dy);
            spatialWeight[(dy + radius) * d + (dx + radius)] = 
                std::exp(dist2 * spaceFactor);
        }
    }
    
    // Color weight factor
    float colorFactor = -0.5f / (sigmaColor * sigmaColor);
    
    std::vector<float> output(width * height);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float centerValue = input[y * width + x];
            float sum = 0.0f;
            float weightSum = 0.0f;
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    // Border handling
                    nx = std::clamp(nx, 0, width - 1);
                    ny = std::clamp(ny, 0, height - 1);
                    
                    float neighborValue = input[ny * width + nx];
                    float colorDiff = neighborValue - centerValue;
                    
                    // Combined weight
                    float weight = spatialWeight[(dy + radius) * d + (dx + radius)] *
                                  std::exp(colorDiff * colorDiff * colorFactor);
                    
                    sum += neighborValue * weight;
                    weightSum += weight;
                }
            }
            
            output[y * width + x] = (weightSum > 0) ? (sum / weightSum) : centerValue;
        }
    }
    
    return output;
}

/**
 * @brief Bilateral filter for color images (3 channels)
 */
inline std::vector<float> bilateralFilterColor(
    const float* input, int width, int height,
    int d = -1,
    float sigmaColor = 75.0f,
    float sigmaSpace = 75.0f
) {
    if (d <= 0) {
        d = static_cast<int>(std::ceil(sigmaSpace * 3) * 2 + 1);
    }
    if (d % 2 == 0) d += 1;
    
    int radius = d / 2;
    
    // Precompute spatial weights
    std::vector<float> spatialWeight(d * d);
    float spaceFactor = -0.5f / (sigmaSpace * sigmaSpace);
    
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            float dist2 = static_cast<float>(dx * dx + dy * dy);
            spatialWeight[(dy + radius) * d + (dx + radius)] = 
                std::exp(dist2 * spaceFactor);
        }
    }
    
    float colorFactor = -0.5f / (sigmaColor * sigmaColor);
    
    std::vector<float> output(width * height * 3);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int centerIdx = (y * width + x) * 3;
            float centerR = input[centerIdx];
            float centerG = input[centerIdx + 1];
            float centerB = input[centerIdx + 2];
            
            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
            float weightSum = 0.0f;
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = std::clamp(x + dx, 0, width - 1);
                    int ny = std::clamp(y + dy, 0, height - 1);
                    
                    int neighborIdx = (ny * width + nx) * 3;
                    float nR = input[neighborIdx];
                    float nG = input[neighborIdx + 1];
                    float nB = input[neighborIdx + 2];
                    
                    // Color difference (Euclidean in color space)
                    float colorDist2 = (nR - centerR) * (nR - centerR) +
                                      (nG - centerG) * (nG - centerG) +
                                      (nB - centerB) * (nB - centerB);
                    
                    float weight = spatialWeight[(dy + radius) * d + (dx + radius)] *
                                  std::exp(colorDist2 * colorFactor);
                    
                    sumR += nR * weight;
                    sumG += nG * weight;
                    sumB += nB * weight;
                    weightSum += weight;
                }
            }
            
            int outIdx = (y * width + x) * 3;
            if (weightSum > 0) {
                output[outIdx] = sumR / weightSum;
                output[outIdx + 1] = sumG / weightSum;
                output[outIdx + 2] = sumB / weightSum;
            } else {
                output[outIdx] = centerR;
                output[outIdx + 1] = centerG;
                output[outIdx + 2] = centerB;
            }
        }
    }
    
    return output;
}

/**
 * @brief Box filter (unnormalized)
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @param ksize Kernel size
 * @param normalize Whether to normalize
 * @param borderType Border handling
 * @return Filtered image
 */
inline std::vector<float> boxFilter(
    const float* input, int width, int height,
    int ksize = 3,
    bool normalize = true,
    int borderType = BORDER_REFLECT_101
) {
    std::vector<float> output(width * height);
    int pad = ksize / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            int count = 0;
            
            for (int ky = -pad; ky <= pad; ++ky) {
                for (int kx = -pad; kx <= pad; ++kx) {
                    int nx = std::clamp(x + kx, 0, width - 1);
                    int ny = std::clamp(y + ky, 0, height - 1);
                    sum += input[ny * width + nx];
                    ++count;
                }
            }
            
            output[y * width + x] = normalize ? (sum / count) : sum;
        }
    }
    
    return output;
}

/**
 * @brief Squared box filter (for variance computation)
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @param ksize Kernel size
 * @param normalize Whether to normalize
 * @return Filtered image (sum of squares)
 */
inline std::vector<float> sqrBoxFilter(
    const float* input, int width, int height,
    int ksize = 3,
    bool normalize = true
) {
    std::vector<float> output(width * height);
    int pad = ksize / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            int count = 0;
            
            for (int ky = -pad; ky <= pad; ++ky) {
                for (int kx = -pad; kx <= pad; ++kx) {
                    int nx = std::clamp(x + kx, 0, width - 1);
                    int ny = std::clamp(y + ky, 0, height - 1);
                    float val = input[ny * width + nx];
                    sum += val * val;
                    ++count;
                }
            }
            
            output[y * width + x] = normalize ? (sum / count) : sum;
        }
    }
    
    return output;
}

/**
 * @brief Joint bilateral filter
 * Uses a guidance image for edge-preserving filtering
 */
inline std::vector<float> jointBilateralFilter(
    const float* input, int width, int height,
    const float* guide,
    int d = -1,
    float sigmaColor = 75.0f,
    float sigmaSpace = 75.0f
) {
    if (d <= 0) {
        d = static_cast<int>(std::ceil(sigmaSpace * 3) * 2 + 1);
    }
    if (d % 2 == 0) d += 1;
    
    int radius = d / 2;
    
    std::vector<float> spatialWeight(d * d);
    float spaceFactor = -0.5f / (sigmaSpace * sigmaSpace);
    
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            float dist2 = static_cast<float>(dx * dx + dy * dy);
            spatialWeight[(dy + radius) * d + (dx + radius)] = 
                std::exp(dist2 * spaceFactor);
        }
    }
    
    float colorFactor = -0.5f / (sigmaColor * sigmaColor);
    
    std::vector<float> output(width * height);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float centerGuide = guide[y * width + x];
            float sum = 0.0f;
            float weightSum = 0.0f;
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = std::clamp(x + dx, 0, width - 1);
                    int ny = std::clamp(y + dy, 0, height - 1);
                    
                    float neighborValue = input[ny * width + nx];
                    float neighborGuide = guide[ny * width + nx];
                    float guideDiff = neighborGuide - centerGuide;
                    
                    float weight = spatialWeight[(dy + radius) * d + (dx + radius)] *
                                  std::exp(guideDiff * guideDiff * colorFactor);
                    
                    sum += neighborValue * weight;
                    weightSum += weight;
                }
            }
            
            output[y * width + x] = (weightSum > 0) ? (sum / weightSum) : input[y * width + x];
        }
    }
    
    return output;
}

/**
 * @brief Guided filter (edge-preserving, fast)
 * @param input Input image
 * @param guide Guide image
 * @param width Image width
 * @param height Image height
 * @param radius Filter radius
 * @param eps Regularization parameter
 * @return Filtered image
 */
inline std::vector<float> guidedFilter(
    const float* input, int width, int height,
    const float* guide,
    int radius = 8,
    float eps = 0.01f
) {
    int ksize = 2 * radius + 1;
    
    // Compute means
    auto meanI = boxFilter(guide, width, height, ksize, true);
    auto meanP = boxFilter(input, width, height, ksize, true);
    
    // Compute correlations
    std::vector<float> Ip(width * height);
    std::vector<float> II(width * height);
    for (int i = 0; i < width * height; ++i) {
        Ip[i] = guide[i] * input[i];
        II[i] = guide[i] * guide[i];
    }
    
    auto meanIp = boxFilter(Ip.data(), width, height, ksize, true);
    auto meanII = boxFilter(II.data(), width, height, ksize, true);
    
    // Compute coefficients
    std::vector<float> a(width * height);
    std::vector<float> b(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        float covIp = meanIp[i] - meanI[i] * meanP[i];
        float varI = meanII[i] - meanI[i] * meanI[i];
        a[i] = covIp / (varI + eps);
        b[i] = meanP[i] - a[i] * meanI[i];
    }
    
    // Compute output
    auto meanA = boxFilter(a.data(), width, height, ksize, true);
    auto meanB = boxFilter(b.data(), width, height, ksize, true);
    
    std::vector<float> output(width * height);
    for (int i = 0; i < width * height; ++i) {
        output[i] = meanA[i] * guide[i] + meanB[i];
    }
    
    return output;
}

/**
 * @brief Self-guided filter (guide = input)
 */
inline std::vector<float> guidedFilter(
    const float* input, int width, int height,
    int radius = 8,
    float eps = 0.01f
) {
    return guidedFilter(input, width, height, input, radius, eps);
}

/**
 * @brief Non-local means denoising
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @param h Filter strength (higher = more smoothing)
 * @param templateWindowSize Size of template patch
 * @param searchWindowSize Size of search area
 * @return Denoised image
 */
inline std::vector<float> fastNlMeansDenoising(
    const float* input, int width, int height,
    float h = 10.0f,
    int templateWindowSize = 7,
    int searchWindowSize = 21
) {
    std::vector<float> output(width * height);
    
    int templateRadius = templateWindowSize / 2;
    int searchRadius = searchWindowSize / 2;
    float h2 = h * h;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            float weightSum = 0.0f;
            
            // Search in neighborhood
            for (int sy = -searchRadius; sy <= searchRadius; ++sy) {
                for (int sx = -searchRadius; sx <= searchRadius; ++sx) {
                    int ny = y + sy;
                    int nx = x + sx;
                    
                    if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
                    
                    // Compute patch distance
                    float dist = 0.0f;
                    int count = 0;
                    
                    for (int ty = -templateRadius; ty <= templateRadius; ++ty) {
                        for (int tx = -templateRadius; tx <= templateRadius; ++tx) {
                            int py1 = std::clamp(y + ty, 0, height - 1);
                            int px1 = std::clamp(x + tx, 0, width - 1);
                            int py2 = std::clamp(ny + ty, 0, height - 1);
                            int px2 = std::clamp(nx + tx, 0, width - 1);
                            
                            float diff = input[py1 * width + px1] - input[py2 * width + px2];
                            dist += diff * diff;
                            ++count;
                        }
                    }
                    
                    dist /= count;
                    float weight = std::exp(-dist / h2);
                    
                    sum += input[ny * width + nx] * weight;
                    weightSum += weight;
                }
            }
            
            output[y * width + x] = (weightSum > 0) ? (sum / weightSum) : input[y * width + x];
        }
    }
    
    return output;
}

/**
 * @brief Anisotropic diffusion (Perona-Malik)
 * Edge-preserving smoothing through diffusion
 */
inline std::vector<float> anisotropicDiffusion(
    const float* input, int width, int height,
    int iterations = 10,
    float kappa = 50.0f,
    float lambda = 0.25f,
    int option = 1  // 1 or 2 for different conductance functions
) {
    std::vector<float> current(input, input + width * height);
    std::vector<float> next(width * height);
    
    for (int iter = 0; iter < iterations; ++iter) {
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int idx = y * width + x;
                
                // Compute gradients in 4 directions
                float dN = current[(y - 1) * width + x] - current[idx];
                float dS = current[(y + 1) * width + x] - current[idx];
                float dE = current[y * width + (x + 1)] - current[idx];
                float dW = current[y * width + (x - 1)] - current[idx];
                
                // Conductance coefficients
                float cN, cS, cE, cW;
                if (option == 1) {
                    cN = std::exp(-(dN * dN) / (kappa * kappa));
                    cS = std::exp(-(dS * dS) / (kappa * kappa));
                    cE = std::exp(-(dE * dE) / (kappa * kappa));
                    cW = std::exp(-(dW * dW) / (kappa * kappa));
                } else {
                    cN = 1.0f / (1.0f + (dN * dN) / (kappa * kappa));
                    cS = 1.0f / (1.0f + (dS * dS) / (kappa * kappa));
                    cE = 1.0f / (1.0f + (dE * dE) / (kappa * kappa));
                    cW = 1.0f / (1.0f + (dW * dW) / (kappa * kappa));
                }
                
                next[idx] = current[idx] + lambda * (cN * dN + cS * dS + cE * dE + cW * dW);
            }
        }
        
        // Copy borders
        for (int x = 0; x < width; ++x) {
            next[x] = current[x];
            next[(height - 1) * width + x] = current[(height - 1) * width + x];
        }
        for (int y = 0; y < height; ++y) {
            next[y * width] = current[y * width];
            next[y * width + (width - 1)] = current[y * width + (width - 1)];
        }
        
        std::swap(current, next);
    }
    
    return current;
}

} // namespace filters
} // namespace neurova

#endif // NEUROVA_FILTERS_BILATERAL_HPP
