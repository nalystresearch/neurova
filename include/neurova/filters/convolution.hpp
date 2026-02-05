// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file convolution.hpp
 * @brief 2D convolution operations for image filtering
 */

#ifndef NEUROVA_FILTERS_CONVOLUTION_HPP
#define NEUROVA_FILTERS_CONVOLUTION_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace neurova {
namespace filters {

// Border types
constexpr int BORDER_CONSTANT = 0;
constexpr int BORDER_REPLICATE = 1;
constexpr int BORDER_REFLECT = 2;
constexpr int BORDER_WRAP = 3;
constexpr int BORDER_REFLECT_101 = 4;
constexpr int BORDER_DEFAULT = BORDER_REFLECT_101;

/**
 * @brief Get pixel value with border handling
 */
inline float getBorderPixel(const float* image, int width, int height,
                            int x, int y, int borderType, float borderValue = 0.0f) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        return image[y * width + x];
    }
    
    switch (borderType) {
        case BORDER_CONSTANT:
            return borderValue;
        case BORDER_REPLICATE:
            x = std::clamp(x, 0, width - 1);
            y = std::clamp(y, 0, height - 1);
            return image[y * width + x];
        case BORDER_REFLECT:
            if (x < 0) x = -x - 1;
            if (x >= width) x = 2 * width - x - 1;
            if (y < 0) y = -y - 1;
            if (y >= height) y = 2 * height - y - 1;
            x = std::clamp(x, 0, width - 1);
            y = std::clamp(y, 0, height - 1);
            return image[y * width + x];
        case BORDER_WRAP:
            x = ((x % width) + width) % width;
            y = ((y % height) + height) % height;
            return image[y * width + x];
        case BORDER_REFLECT_101:
        default:
            if (x < 0) x = -x;
            if (x >= width) x = 2 * width - x - 2;
            if (y < 0) y = -y;
            if (y >= height) y = 2 * height - y - 2;
            x = std::clamp(x, 0, width - 1);
            y = std::clamp(y, 0, height - 1);
            return image[y * width + x];
    }
}

/**
 * @brief 2D convolution operation
 * @param input Input image (grayscale, HxW)
 * @param width Image width
 * @param height Image height
 * @param kernel Convolution kernel (flattened kH x kW)
 * @param kWidth Kernel width
 * @param kHeight Kernel height
 * @param borderType Border handling type
 * @param borderValue Value for BORDER_CONSTANT
 * @return Convolved image
 */
inline std::vector<float> convolve2d(
    const float* input, int width, int height,
    const float* kernel, int kWidth, int kHeight,
    int borderType = BORDER_REFLECT_101,
    float borderValue = 0.0f
) {
    std::vector<float> output(width * height);
    
    int padX = kWidth / 2;
    int padY = kHeight / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            
            for (int ky = 0; ky < kHeight; ++ky) {
                for (int kx = 0; kx < kWidth; ++kx) {
                    int ix = x + kx - padX;
                    int iy = y + ky - padY;
                    
                    float pixel = getBorderPixel(input, width, height, 
                                                 ix, iy, borderType, borderValue);
                    // Note: standard uses flipped kernel
                    int ki = (kHeight - 1 - ky) * kWidth + (kWidth - 1 - kx);
                    sum += pixel * kernel[ki];
                }
            }
            
            output[y * width + x] = sum;
        }
    }
    
    return output;
}

/**
 * @brief 2D convolution with vector inputs
 */
inline std::vector<float> convolve2d(
    const std::vector<float>& input, int width, int height,
    const std::vector<float>& kernel, int kWidth, int kHeight,
    int borderType = BORDER_REFLECT_101,
    float borderValue = 0.0f
) {
    return convolve2d(input.data(), width, height,
                     kernel.data(), kWidth, kHeight,
                     borderType, borderValue);
}

/**
 * @brief Apply 2D filter (correlation, not convolution - kernel not flipped)
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @param kernel Filter kernel
 * @param kWidth Kernel width
 * @param kHeight Kernel height
 * @param anchor Anchor point (-1,-1 for center)
 * @param delta Value added to filtered pixels
 * @param borderType Border handling
 * @return Filtered image
 */
inline std::vector<float> filter2D(
    const float* input, int width, int height,
    const float* kernel, int kWidth, int kHeight,
    int anchorX = -1, int anchorY = -1,
    float delta = 0.0f,
    int borderType = BORDER_REFLECT_101
) {
    std::vector<float> output(width * height);
    
    if (anchorX < 0) anchorX = kWidth / 2;
    if (anchorY < 0) anchorY = kHeight / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            
            for (int ky = 0; ky < kHeight; ++ky) {
                for (int kx = 0; kx < kWidth; ++kx) {
                    int ix = x + kx - anchorX;
                    int iy = y + ky - anchorY;
                    
                    float pixel = getBorderPixel(input, width, height, 
                                                 ix, iy, borderType, 0.0f);
                    sum += pixel * kernel[ky * kWidth + kx];
                }
            }
            
            output[y * width + x] = sum + delta;
        }
    }
    
    return output;
}

/**
 * @brief Separable 2D convolution (faster for separable kernels)
 * @param input Input image
 * @param width Image width
 * @param height Image height
 * @param kernelX Horizontal kernel (1D)
 * @param kernelY Vertical kernel (1D)
 * @param kSize Kernel size
 * @param borderType Border handling
 * @return Filtered image
 */
inline std::vector<float> sepFilter2D(
    const float* input, int width, int height,
    const float* kernelX, const float* kernelY, int kSize,
    int borderType = BORDER_REFLECT_101
) {
    std::vector<float> temp(width * height);
    std::vector<float> output(width * height);
    
    int pad = kSize / 2;
    
    // Horizontal pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int k = 0; k < kSize; ++k) {
                int ix = x + k - pad;
                float pixel = getBorderPixel(input, width, height, 
                                             ix, y, borderType, 0.0f);
                sum += pixel * kernelX[k];
            }
            temp[y * width + x] = sum;
        }
    }
    
    // Vertical pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int k = 0; k < kSize; ++k) {
                int iy = y + k - pad;
                float pixel = getBorderPixel(temp.data(), width, height, 
                                             x, iy, borderType, 0.0f);
                sum += pixel * kernelY[k];
            }
            output[y * width + x] = sum;
        }
    }
    
    return output;
}

/**
 * @brief Apply morphological operation
 * @param input Input binary image
 * @param width Image width
 * @param height Image height
 * @param kernel Structuring element
 * @param kWidth Kernel width
 * @param kHeight Kernel height
 * @param op Operation: 0=ERODE, 1=DILATE
 * @return Result image
 */
inline std::vector<float> morphologyOp(
    const float* input, int width, int height,
    const uint8_t* kernel, int kWidth, int kHeight,
    int op
) {
    std::vector<float> output(width * height);
    
    int padX = kWidth / 2;
    int padY = kHeight / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float result = (op == 0) ? 255.0f : 0.0f; // ERODE: min, DILATE: max
            
            for (int ky = 0; ky < kHeight; ++ky) {
                for (int kx = 0; kx < kWidth; ++kx) {
                    if (!kernel[ky * kWidth + kx]) continue;
                    
                    int ix = x + kx - padX;
                    int iy = y + ky - padY;
                    
                    if (ix < 0 || ix >= width || iy < 0 || iy >= height) {
                        if (op == 0) result = 0.0f; // Erode boundary
                        continue;
                    }
                    
                    float pixel = input[iy * width + ix];
                    if (op == 0) {
                        result = std::min(result, pixel);
                    } else {
                        result = std::max(result, pixel);
                    }
                }
            }
            
            output[y * width + x] = result;
        }
    }
    
    return output;
}

/**
 * @brief Erode an image
 */
inline std::vector<float> erode(
    const float* input, int width, int height,
    const uint8_t* kernel, int kWidth, int kHeight,
    int iterations = 1
) {
    std::vector<float> result(input, input + width * height);
    
    for (int i = 0; i < iterations; ++i) {
        result = morphologyOp(result.data(), width, height,
                             kernel, kWidth, kHeight, 0);
    }
    
    return result;
}

/**
 * @brief Dilate an image
 */
inline std::vector<float> dilate(
    const float* input, int width, int height,
    const uint8_t* kernel, int kWidth, int kHeight,
    int iterations = 1
) {
    std::vector<float> result(input, input + width * height);
    
    for (int i = 0; i < iterations; ++i) {
        result = morphologyOp(result.data(), width, height,
                             kernel, kWidth, kHeight, 1);
    }
    
    return result;
}

/**
 * @brief Morphological opening (erode then dilate)
 */
inline std::vector<float> morphOpen(
    const float* input, int width, int height,
    const uint8_t* kernel, int kWidth, int kHeight,
    int iterations = 1
) {
    auto result = erode(input, width, height, kernel, kWidth, kHeight, iterations);
    return dilate(result.data(), width, height, kernel, kWidth, kHeight, iterations);
}

/**
 * @brief Morphological closing (dilate then erode)
 */
inline std::vector<float> morphClose(
    const float* input, int width, int height,
    const uint8_t* kernel, int kWidth, int kHeight,
    int iterations = 1
) {
    auto result = dilate(input, width, height, kernel, kWidth, kHeight, iterations);
    return erode(result.data(), width, height, kernel, kWidth, kHeight, iterations);
}

/**
 * @brief Morphological gradient (dilate - erode)
 */
inline std::vector<float> morphGradient(
    const float* input, int width, int height,
    const uint8_t* kernel, int kWidth, int kHeight
) {
    auto dilated = dilate(input, width, height, kernel, kWidth, kHeight);
    auto eroded = erode(input, width, height, kernel, kWidth, kHeight);
    
    std::vector<float> result(width * height);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = dilated[i] - eroded[i];
    }
    
    return result;
}

/**
 * @brief Top hat transform (input - opening)
 */
inline std::vector<float> topHat(
    const float* input, int width, int height,
    const uint8_t* kernel, int kWidth, int kHeight
) {
    auto opened = morphOpen(input, width, height, kernel, kWidth, kHeight);
    
    std::vector<float> result(width * height);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = input[i] - opened[i];
    }
    
    return result;
}

/**
 * @brief Black hat transform (closing - input)
 */
inline std::vector<float> blackHat(
    const float* input, int width, int height,
    const uint8_t* kernel, int kWidth, int kHeight
) {
    auto closed = morphClose(input, width, height, kernel, kWidth, kHeight);
    
    std::vector<float> result(width * height);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = closed[i] - input[i];
    }
    
    return result;
}

// Morphology operation constants
constexpr int MORPH_ERODE = 0;
constexpr int MORPH_DILATE = 1;
constexpr int MORPH_OPEN = 2;
constexpr int MORPH_CLOSE = 3;
constexpr int MORPH_GRADIENT = 4;
constexpr int MORPH_TOPHAT = 5;
constexpr int MORPH_BLACKHAT = 6;

/**
 * @brief General morphological operation
 */
inline std::vector<float> morphologyEx(
    const float* input, int width, int height,
    int op,
    const uint8_t* kernel, int kWidth, int kHeight,
    int iterations = 1
) {
    switch (op) {
        case MORPH_ERODE:
            return erode(input, width, height, kernel, kWidth, kHeight, iterations);
        case MORPH_DILATE:
            return dilate(input, width, height, kernel, kWidth, kHeight, iterations);
        case MORPH_OPEN:
            return morphOpen(input, width, height, kernel, kWidth, kHeight, iterations);
        case MORPH_CLOSE:
            return morphClose(input, width, height, kernel, kWidth, kHeight, iterations);
        case MORPH_GRADIENT:
            return morphGradient(input, width, height, kernel, kWidth, kHeight);
        case MORPH_TOPHAT:
            return topHat(input, width, height, kernel, kWidth, kHeight);
        case MORPH_BLACKHAT:
            return blackHat(input, width, height, kernel, kWidth, kHeight);
        default:
            return erode(input, width, height, kernel, kWidth, kHeight, iterations);
    }
}

} // namespace filters
} // namespace neurova

#endif // NEUROVA_FILTERS_CONVOLUTION_HPP
