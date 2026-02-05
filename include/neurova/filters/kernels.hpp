// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file kernels.hpp
 * @brief Kernel generation functions for image filtering
 */

#ifndef NEUROVA_FILTERS_KERNELS_HPP
#define NEUROVA_FILTERS_KERNELS_HPP

#include <vector>
#include <cmath>
#include <algorithm>

namespace neurova {
namespace filters {

/**
 * @brief Create a box (averaging) kernel
 * @param size Kernel size (width and height)
 * @param normalize If true, normalize kernel values
 * @return Flattened kernel as vector
 */
inline std::vector<float> boxKernel(int size, bool normalize = true) {
    std::vector<float> kernel(size * size);
    float value = normalize ? 1.0f / (size * size) : 1.0f;
    std::fill(kernel.begin(), kernel.end(), value);
    return kernel;
}

/**
 * @brief Create a Gaussian kernel
 * @param size Kernel size (must be odd)
 * @param sigma Standard deviation (0 = auto-compute)
 * @return Flattened kernel as vector
 */
inline std::vector<float> gaussianKernel(int size, float sigma = 0.0f) {
    if (size % 2 == 0) {
        size += 1;
    }
    
    if (sigma <= 0) {
        sigma = 0.3f * ((size - 1) * 0.5f - 1) + 0.8f;
    }
    
    std::vector<float> kernel(size * size);
    int center = size / 2;
    float sum = 0.0f;
    
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float dx = static_cast<float>(x - center);
            float dy = static_cast<float>(y - center);
            float value = std::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            kernel[y * size + x] = value;
            sum += value;
        }
    }
    
    // Normalize
    for (auto& v : kernel) {
        v /= sum;
    }
    
    return kernel;
}

/**
 * @brief Get a 1D Gaussian kernel
 * @param size Kernel size
 * @param sigma Standard deviation
 * @return 1D Gaussian kernel
 */
inline std::vector<float> getGaussianKernel(int size, float sigma) {
    if (sigma <= 0) {
        sigma = 0.3f * ((size - 1) * 0.5f - 1) + 0.8f;
    }
    
    std::vector<float> kernel(size);
    int center = size / 2;
    float sum = 0.0f;
    
    for (int i = 0; i < size; ++i) {
        float x = static_cast<float>(i - center);
        kernel[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }
    
    for (auto& v : kernel) {
        v /= sum;
    }
    
    return kernel;
}

/**
 * @brief Get Sobel kernels for gradient computation
 * @param ksize Kernel size (1, 3, 5, or 7)
 * @return Pair of (Gx kernel, Gy kernel)
 */
inline std::pair<std::vector<float>, std::vector<float>> sobelKernels(int ksize = 3) {
    std::vector<float> gx, gy;
    
    if (ksize == 1) {
        gx = {-1.0f, 0.0f, 1.0f};
        gy = {-1.0f, 0.0f, 1.0f};
    } else if (ksize == 3) {
        gx = {
            -1.0f, 0.0f, 1.0f,
            -2.0f, 0.0f, 2.0f,
            -1.0f, 0.0f, 1.0f
        };
        gy = {
            -1.0f, -2.0f, -1.0f,
             0.0f,  0.0f,  0.0f,
             1.0f,  2.0f,  1.0f
        };
    } else if (ksize == 5) {
        gx = {
            -1.0f, -2.0f, 0.0f, 2.0f, 1.0f,
            -4.0f, -8.0f, 0.0f, 8.0f, 4.0f,
            -6.0f, -12.0f, 0.0f, 12.0f, 6.0f,
            -4.0f, -8.0f, 0.0f, 8.0f, 4.0f,
            -1.0f, -2.0f, 0.0f, 2.0f, 1.0f
        };
        gy = {
            -1.0f, -4.0f, -6.0f, -4.0f, -1.0f,
            -2.0f, -8.0f, -12.0f, -8.0f, -2.0f,
             0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
             2.0f,  8.0f, 12.0f,  8.0f,  2.0f,
             1.0f,  4.0f,  6.0f,  4.0f,  1.0f
        };
    } else if (ksize == 7) {
        gx = {
            -1.0f, -4.0f, -5.0f, 0.0f, 5.0f, 4.0f, 1.0f,
            -6.0f, -24.0f, -30.0f, 0.0f, 30.0f, 24.0f, 6.0f,
            -15.0f, -60.0f, -75.0f, 0.0f, 75.0f, 60.0f, 15.0f,
            -20.0f, -80.0f, -100.0f, 0.0f, 100.0f, 80.0f, 20.0f,
            -15.0f, -60.0f, -75.0f, 0.0f, 75.0f, 60.0f, 15.0f,
            -6.0f, -24.0f, -30.0f, 0.0f, 30.0f, 24.0f, 6.0f,
            -1.0f, -4.0f, -5.0f, 0.0f, 5.0f, 4.0f, 1.0f
        };
        // Transpose for gy
        gy.resize(49);
        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j < 7; ++j) {
                gy[i * 7 + j] = gx[j * 7 + i];
            }
        }
    } else {
        // Default to 3x3
        return sobelKernels(3);
    }
    
    return {gx, gy};
}

/**
 * @brief Get Scharr kernels (more accurate than Sobel for 3x3)
 * @return Pair of (Gx kernel, Gy kernel)
 */
inline std::pair<std::vector<float>, std::vector<float>> scharrKernels() {
    std::vector<float> gx = {
        -3.0f, 0.0f, 3.0f,
        -10.0f, 0.0f, 10.0f,
        -3.0f, 0.0f, 3.0f
    };
    std::vector<float> gy = {
        -3.0f, -10.0f, -3.0f,
         0.0f,   0.0f,  0.0f,
         3.0f,  10.0f,  3.0f
    };
    return {gx, gy};
}

/**
 * @brief Get Laplacian kernel
 * @param ksize Kernel size (1, 3, 5, or 7)
 * @return Laplacian kernel
 */
inline std::vector<float> laplacianKernel(int ksize = 3) {
    if (ksize == 1) {
        return {
            0.0f, 1.0f, 0.0f,
            1.0f, -4.0f, 1.0f,
            0.0f, 1.0f, 0.0f
        };
    } else if (ksize == 3) {
        return {
            0.0f, 1.0f, 0.0f,
            1.0f, -4.0f, 1.0f,
            0.0f, 1.0f, 0.0f
        };
    } else if (ksize == 5) {
        return {
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 2.0f, 1.0f, 0.0f,
            1.0f, 2.0f, -16.0f, 2.0f, 1.0f,
            0.0f, 1.0f, 2.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f
        };
    } else if (ksize == 7) {
        return {
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 2.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 0.0f,
            1.0f, 2.0f, 3.0f, -44.0f, 3.0f, 2.0f, 1.0f,
            0.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 2.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f
        };
    }
    return laplacianKernel(3);
}

/**
 * @brief Get sharpening kernel
 * @param strength Sharpening strength (default 1.0)
 * @return Sharpening kernel
 */
inline std::vector<float> sharpenKernel(float strength = 1.0f) {
    return {
        0.0f, -strength, 0.0f,
        -strength, 1.0f + 4.0f * strength, -strength,
        0.0f, -strength, 0.0f
    };
}

/**
 * @brief Get emboss kernel
 * @return Emboss kernel
 */
inline std::vector<float> embossKernel() {
    return {
        -2.0f, -1.0f, 0.0f,
        -1.0f, 1.0f, 1.0f,
        0.0f, 1.0f, 2.0f
    };
}

/**
 * @brief Get edge detection kernel
 * @return Edge detection kernel
 */
inline std::vector<float> edgeKernel() {
    return {
        -1.0f, -1.0f, -1.0f,
        -1.0f, 8.0f, -1.0f,
        -1.0f, -1.0f, -1.0f
    };
}

/**
 * @brief Get Prewitt kernels for gradient computation
 * @return Pair of (Gx kernel, Gy kernel)
 */
inline std::pair<std::vector<float>, std::vector<float>> prewittKernels() {
    std::vector<float> gx = {
        -1.0f, 0.0f, 1.0f,
        -1.0f, 0.0f, 1.0f,
        -1.0f, 0.0f, 1.0f
    };
    std::vector<float> gy = {
        -1.0f, -1.0f, -1.0f,
         0.0f,  0.0f,  0.0f,
         1.0f,  1.0f,  1.0f
    };
    return {gx, gy};
}

/**
 * @brief Get Roberts cross kernels
 * @return Pair of (Gx kernel, Gy kernel)
 */
inline std::pair<std::vector<float>, std::vector<float>> robertsKernels() {
    std::vector<float> gx = {
        1.0f, 0.0f,
        0.0f, -1.0f
    };
    std::vector<float> gy = {
        0.0f, 1.0f,
        -1.0f, 0.0f
    };
    return {gx, gy};
}

/**
 * @brief Get unsharp masking kernel
 * @param amount Sharpening amount
 * @param sigma Gaussian sigma for blur estimation
 * @return Unsharp mask kernel (for direct convolution)
 */
inline std::vector<float> unsharpMaskKernel(float amount = 1.0f, int size = 5) {
    auto gaussian = gaussianKernel(size, 0);
    std::vector<float> kernel(size * size);
    
    int center = (size * size) / 2;
    for (size_t i = 0; i < kernel.size(); ++i) {
        kernel[i] = -amount * gaussian[i];
    }
    kernel[center] += 1.0f + amount;
    
    return kernel;
}

/**
 * @brief Get LoG (Laplacian of Gaussian) kernel
 * @param size Kernel size
 * @param sigma Standard deviation
 * @return LoG kernel
 */
inline std::vector<float> logKernel(int size, float sigma) {
    if (sigma <= 0) {
        sigma = 0.3f * ((size - 1) * 0.5f - 1) + 0.8f;
    }
    
    std::vector<float> kernel(size * size);
    int center = size / 2;
    float sigma2 = sigma * sigma;
    float sigma4 = sigma2 * sigma2;
    
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float dx = static_cast<float>(x - center);
            float dy = static_cast<float>(y - center);
            float r2 = dx * dx + dy * dy;
            kernel[y * size + x] = (r2 - 2.0f * sigma2) / sigma4 * 
                                   std::exp(-r2 / (2.0f * sigma2));
        }
    }
    
    // Normalize to sum to 0
    float sum = 0.0f;
    for (auto v : kernel) sum += v;
    float mean = sum / (size * size);
    for (auto& v : kernel) v -= mean;
    
    return kernel;
}

/**
 * @brief Get DoG (Difference of Gaussians) kernel
 * @param size Kernel size
 * @param sigma1 First Gaussian sigma
 * @param sigma2 Second Gaussian sigma
 * @return DoG kernel
 */
inline std::vector<float> dogKernel(int size, float sigma1, float sigma2) {
    auto g1 = gaussianKernel(size, sigma1);
    auto g2 = gaussianKernel(size, sigma2);
    
    std::vector<float> kernel(size * size);
    for (size_t i = 0; i < kernel.size(); ++i) {
        kernel[i] = g1[i] - g2[i];
    }
    
    return kernel;
}

/**
 * @brief Create a morphological structuring element
 * @param shape Shape type (0=RECT, 1=CROSS, 2=ELLIPSE)
 * @param size Size of structuring element
 * @return Structuring element as binary vector
 */
inline std::vector<uint8_t> getStructuringElement(int shape, int sizeX, int sizeY = -1) {
    if (sizeY < 0) sizeY = sizeX;
    
    std::vector<uint8_t> kernel(sizeX * sizeY, 0);
    int cx = sizeX / 2;
    int cy = sizeY / 2;
    
    if (shape == 0) { // RECT
        std::fill(kernel.begin(), kernel.end(), 1);
    } else if (shape == 1) { // CROSS
        for (int y = 0; y < sizeY; ++y) {
            kernel[y * sizeX + cx] = 1;
        }
        for (int x = 0; x < sizeX; ++x) {
            kernel[cy * sizeX + x] = 1;
        }
    } else if (shape == 2) { // ELLIPSE
        float rx = sizeX / 2.0f;
        float ry = sizeY / 2.0f;
        for (int y = 0; y < sizeY; ++y) {
            for (int x = 0; x < sizeX; ++x) {
                float dx = (x - cx) / rx;
                float dy = (y - cy) / ry;
                if (dx * dx + dy * dy <= 1.0f) {
                    kernel[y * sizeX + x] = 1;
                }
            }
        }
    }
    
    return kernel;
}

// Structuring element shape constants
constexpr int MORPH_RECT = 0;
constexpr int MORPH_CROSS = 1;
constexpr int MORPH_ELLIPSE = 2;

} // namespace filters
} // namespace neurova

#endif // NEUROVA_FILTERS_KERNELS_HPP
