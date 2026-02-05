// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file blur.hpp
 * @brief Image blurring and smoothing filters
 */

#ifndef NEUROVA_FILTERS_BLUR_HPP
#define NEUROVA_FILTERS_BLUR_HPP

#include "kernels.hpp"
#include "convolution.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace neurova {
namespace filters {

/**
 * @brief Box blur (uniform averaging filter)
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @param ksize Kernel size (odd number)
 * @param normalize Whether to normalize the kernel
 * @param borderType Border handling type
 * @return Blurred image
 */
inline std::vector<float> boxBlur(
    const float* input, int width, int height,
    int ksize = 3,
    bool normalize = true,
    int borderType = BORDER_REFLECT_101
) {
    if (ksize % 2 == 0) ksize += 1;
    
    auto kernel = boxKernel(ksize, normalize);
    return convolve2d(input, width, height,
                     kernel.data(), ksize, ksize,
                     borderType);
}

/**
 * @brief Fast box blur using integral image (constant time)
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @param ksize Kernel size
 * @return Blurred image
 */
inline std::vector<float> boxBlurFast(
    const float* input, int width, int height,
    int ksize
) {
    // Compute integral image
    std::vector<double> integral((width + 1) * (height + 1), 0.0);
    
    for (int y = 0; y < height; ++y) {
        double rowSum = 0.0;
        for (int x = 0; x < width; ++x) {
            rowSum += input[y * width + x];
            integral[(y + 1) * (width + 1) + (x + 1)] = 
                rowSum + integral[y * (width + 1) + (x + 1)];
        }
    }
    
    std::vector<float> output(width * height);
    int r = ksize / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int x1 = std::max(0, x - r);
            int y1 = std::max(0, y - r);
            int x2 = std::min(width - 1, x + r);
            int y2 = std::min(height - 1, y + r);
            
            double sum = integral[(y2 + 1) * (width + 1) + (x2 + 1)]
                       - integral[(y1) * (width + 1) + (x2 + 1)]
                       - integral[(y2 + 1) * (width + 1) + (x1)]
                       + integral[(y1) * (width + 1) + (x1)];
            
            int count = (x2 - x1 + 1) * (y2 - y1 + 1);
            output[y * width + x] = static_cast<float>(sum / count);
        }
    }
    
    return output;
}

/**
 * @brief Gaussian blur
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @param ksize Kernel size (odd, or 0 for auto)
 * @param sigmaX Gaussian sigma in X direction
 * @param sigmaY Gaussian sigma in Y direction (0 = same as sigmaX)
 * @param borderType Border handling type
 * @return Blurred image
 */
inline std::vector<float> gaussianBlur(
    const float* input, int width, int height,
    int ksize = 0,
    float sigmaX = 0.0f,
    float sigmaY = 0.0f,
    int borderType = BORDER_REFLECT_101
) {
    // Auto-compute kernel size from sigma if needed
    if (ksize <= 0 && sigmaX > 0) {
        ksize = static_cast<int>(std::ceil(sigmaX * 6)) | 1;
    }
    if (ksize <= 0) ksize = 3;
    if (ksize % 2 == 0) ksize += 1;
    
    if (sigmaX <= 0) {
        sigmaX = 0.3f * ((ksize - 1) * 0.5f - 1) + 0.8f;
    }
    if (sigmaY <= 0) sigmaY = sigmaX;
    
    // Use separable filter for efficiency
    auto kernelX = getGaussianKernel(ksize, sigmaX);
    auto kernelY = getGaussianKernel(ksize, sigmaY);
    
    return sepFilter2D(input, width, height,
                      kernelX.data(), kernelY.data(), ksize,
                      borderType);
}

/**
 * @brief Median blur (non-linear filter)
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @param ksize Kernel size (must be odd)
 * @return Filtered image
 */
inline std::vector<float> medianBlur(
    const float* input, int width, int height,
    int ksize = 3
) {
    if (ksize % 2 == 0) ksize += 1;
    
    std::vector<float> output(width * height);
    int pad = ksize / 2;
    int medianIdx = (ksize * ksize) / 2;
    
    std::vector<float> window(ksize * ksize);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int count = 0;
            
            for (int ky = -pad; ky <= pad; ++ky) {
                for (int kx = -pad; kx <= pad; ++kx) {
                    int ix = std::clamp(x + kx, 0, width - 1);
                    int iy = std::clamp(y + ky, 0, height - 1);
                    window[count++] = input[iy * width + ix];
                }
            }
            
            std::nth_element(window.begin(), 
                           window.begin() + medianIdx, 
                           window.end());
            output[y * width + x] = window[medianIdx];
        }
    }
    
    return output;
}

/**
 * @brief Stack blur (fast approximation to Gaussian)
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @param radius Blur radius
 * @return Blurred image
 */
inline std::vector<float> stackBlur(
    const float* input, int width, int height,
    int radius
) {
    if (radius < 1) return std::vector<float>(input, input + width * height);
    
    std::vector<float> temp(width * height);
    std::vector<float> output(width * height);
    
    int div = radius * 2 + 1;
    
    // Horizontal pass
    for (int y = 0; y < height; ++y) {
        float sum = 0;
        float sumWeighted = 0;
        float weight = 0;
        
        // Initialize
        for (int i = -radius; i <= radius; ++i) {
            int x = std::clamp(i, 0, width - 1);
            float w = static_cast<float>(radius + 1 - std::abs(i));
            sum += input[y * width + x] * w;
            weight += w;
        }
        
        for (int x = 0; x < width; ++x) {
            temp[y * width + x] = sum / weight;
            
            // Slide window
            int leftIdx = std::max(0, x - radius);
            int rightIdx = std::min(width - 1, x + radius + 1);
            
            float leftVal = input[y * width + leftIdx];
            float rightVal = input[y * width + rightIdx];
            
            sum -= leftVal * (radius + 1 - std::abs(x - radius - leftIdx));
            sum += rightVal * (radius + 1 - std::abs(x + radius + 1 - rightIdx));
        }
    }
    
    // Vertical pass
    for (int x = 0; x < width; ++x) {
        float sum = 0;
        float weight = 0;
        
        // Initialize
        for (int i = -radius; i <= radius; ++i) {
            int y = std::clamp(i, 0, height - 1);
            float w = static_cast<float>(radius + 1 - std::abs(i));
            sum += temp[y * width + x] * w;
            weight += w;
        }
        
        for (int y = 0; y < height; ++y) {
            output[y * width + x] = sum / weight;
            
            int topIdx = std::max(0, y - radius);
            int bottomIdx = std::min(height - 1, y + radius + 1);
            
            float topVal = temp[topIdx * width + x];
            float bottomVal = temp[bottomIdx * width + x];
            
            sum -= topVal * (radius + 1 - std::abs(y - radius - topIdx));
            sum += bottomVal * (radius + 1 - std::abs(y + radius + 1 - bottomIdx));
        }
    }
    
    return output;
}

/**
 * @brief Sharpen image
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @param amount Sharpening amount (1.0 = normal)
 * @return Sharpened image
 */
inline std::vector<float> sharpen(
    const float* input, int width, int height,
    float amount = 1.0f
) {
    auto kernel = sharpenKernel(amount);
    return filter2D(input, width, height,
                   kernel.data(), 3, 3);
}

/**
 * @brief Unsharp masking
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @param amount Sharpening amount
 * @param radius Blur radius for mask
 * @param threshold Threshold for sharpening
 * @return Sharpened image
 */
inline std::vector<float> unsharpMask(
    const float* input, int width, int height,
    float amount = 1.0f,
    int radius = 2,
    float threshold = 0.0f
) {
    // Create blurred version
    auto blurred = gaussianBlur(input, width, height, 
                               radius * 2 + 1, static_cast<float>(radius));
    
    std::vector<float> output(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        float diff = input[i] - blurred[i];
        
        if (std::abs(diff) > threshold) {
            output[i] = std::clamp(input[i] + amount * diff, 0.0f, 255.0f);
        } else {
            output[i] = input[i];
        }
    }
    
    return output;
}

/**
 * @brief Motion blur
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @param size Blur size
 * @param angle Blur angle in degrees
 * @return Blurred image
 */
inline std::vector<float> motionBlur(
    const float* input, int width, int height,
    int size = 9,
    float angle = 0.0f
) {
    // Create motion blur kernel
    std::vector<float> kernel(size * size, 0.0f);
    int center = size / 2;
    
    float radians = angle * 3.14159265f / 180.0f;
    float cosA = std::cos(radians);
    float sinA = std::sin(radians);
    
    int count = 0;
    for (int i = 0; i < size; ++i) {
        float t = static_cast<float>(i - center) / center;
        int x = center + static_cast<int>(t * center * cosA);
        int y = center + static_cast<int>(t * center * sinA);
        
        if (x >= 0 && x < size && y >= 0 && y < size) {
            kernel[y * size + x] = 1.0f;
            ++count;
        }
    }
    
    // Normalize
    if (count > 0) {
        for (auto& v : kernel) v /= count;
    }
    
    return filter2D(input, width, height,
                   kernel.data(), size, size);
}

/**
 * @brief Apply blur to color image (3 channels)
 */
inline std::vector<float> gaussianBlurColor(
    const float* input, int width, int height,
    int ksize = 5,
    float sigma = 0.0f
) {
    std::vector<float> output(width * height * 3);
    
    // Split channels
    std::vector<float> channels[3];
    for (int c = 0; c < 3; ++c) {
        channels[c].resize(width * height);
        for (int i = 0; i < width * height; ++i) {
            channels[c][i] = input[i * 3 + c];
        }
    }
    
    // Blur each channel
    for (int c = 0; c < 3; ++c) {
        auto blurred = gaussianBlur(channels[c].data(), width, height,
                                   ksize, sigma);
        for (int i = 0; i < width * height; ++i) {
            output[i * 3 + c] = blurred[i];
        }
    }
    
    return output;
}

/**
 * @brief Blur function matching standard API
 */
inline std::vector<float> blur(
    const float* input, int width, int height,
    int ksizeX, int ksizeY = -1,
    int anchorX = -1, int anchorY = -1,
    int borderType = BORDER_REFLECT_101
) {
    if (ksizeY < 0) ksizeY = ksizeX;
    
    auto kernel = boxKernel(ksizeX, true);
    
    // If rectangular kernel needed
    if (ksizeX != ksizeY) {
        kernel.resize(ksizeX * ksizeY);
        float value = 1.0f / (ksizeX * ksizeY);
        std::fill(kernel.begin(), kernel.end(), value);
    }
    
    return filter2D(input, width, height,
                   kernel.data(), ksizeX, ksizeY,
                   anchorX, anchorY, 0.0f, borderType);
}

/**
 * @brief GaussianBlur with size as pair
 */
inline std::vector<float> GaussianBlur(
    const float* input, int width, int height,
    int ksizeX, int ksizeY,
    float sigmaX, float sigmaY = 0.0f,
    int borderType = BORDER_REFLECT_101
) {
    // For non-square kernels, use 2D Gaussian
    if (ksizeX != ksizeY || sigmaX != sigmaY) {
        auto kernel = gaussianKernel(std::max(ksizeX, ksizeY), sigmaX);
        return convolve2d(input, width, height,
                        kernel.data(), ksizeX, ksizeY,
                        borderType);
    }
    
    return gaussianBlur(input, width, height, ksizeX, sigmaX, sigmaY, borderType);
}

} // namespace filters
} // namespace neurova

#endif // NEUROVA_FILTERS_BLUR_HPP
