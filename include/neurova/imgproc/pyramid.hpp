// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file pyramid.hpp
 * @brief Image pyramid operations
 */

#ifndef NEUROVA_IMGPROC_PYRAMID_HPP
#define NEUROVA_IMGPROC_PYRAMID_HPP

#include <vector>
#include <cmath>
#include <algorithm>

namespace neurova {
namespace imgproc {

/**
 * @brief Downsample image by factor of 2 with Gaussian smoothing
 */
inline std::vector<float> pyrDown(
    const float* src, int srcWidth, int srcHeight, int channels,
    int& dstWidth, int& dstHeight
) {
    dstWidth = (srcWidth + 1) / 2;
    dstHeight = (srcHeight + 1) / 2;
    
    std::vector<float> dst(dstWidth * dstHeight * channels);
    
    // Gaussian kernel for smoothing: [1, 4, 6, 4, 1] / 16
    const float kernel[] = {1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16};
    
    // Apply separable Gaussian filter and downsample
    // First: horizontal pass to temp buffer
    std::vector<float> temp(srcWidth * srcHeight * channels);
    
    for (int y = 0; y < srcHeight; ++y) {
        for (int x = 0; x < srcWidth; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int k = -2; k <= 2; ++k) {
                    int sx = std::max(0, std::min(srcWidth - 1, x + k));
                    sum += kernel[k + 2] * src[(y * srcWidth + sx) * channels + c];
                }
                temp[(y * srcWidth + x) * channels + c] = sum;
            }
        }
    }
    
    // Second: vertical pass and downsample
    for (int y = 0; y < dstHeight; ++y) {
        int srcY = y * 2;
        for (int x = 0; x < dstWidth; ++x) {
            int srcX = x * 2;
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int k = -2; k <= 2; ++k) {
                    int sy = std::max(0, std::min(srcHeight - 1, srcY + k));
                    sum += kernel[k + 2] * temp[(sy * srcWidth + srcX) * channels + c];
                }
                dst[(y * dstWidth + x) * channels + c] = sum;
            }
        }
    }
    
    return dst;
}

/**
 * @brief Upsample image by factor of 2 with interpolation
 */
inline std::vector<float> pyrUp(
    const float* src, int srcWidth, int srcHeight, int channels,
    int& dstWidth, int& dstHeight
) {
    dstWidth = srcWidth * 2;
    dstHeight = srcHeight * 2;
    
    std::vector<float> dst(dstWidth * dstHeight * channels, 0.0f);
    
    // Insert zeros and apply Gaussian filter scaled by 4
    // Kernel: [1, 4, 6, 4, 1] / 8 (doubled for upsampling)
    const float kernel[] = {1.0f/8, 4.0f/8, 6.0f/8, 4.0f/8, 1.0f/8};
    
    // Place source pixels at even positions
    std::vector<float> temp(dstWidth * dstHeight * channels, 0.0f);
    for (int y = 0; y < srcHeight; ++y) {
        for (int x = 0; x < srcWidth; ++x) {
            for (int c = 0; c < channels; ++c) {
                temp[(y * 2 * dstWidth + x * 2) * channels + c] = 
                    4.0f * src[(y * srcWidth + x) * channels + c];
            }
        }
    }
    
    // Horizontal pass
    std::vector<float> temp2(dstWidth * dstHeight * channels, 0.0f);
    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int k = -2; k <= 2; ++k) {
                    int sx = std::max(0, std::min(dstWidth - 1, x + k));
                    sum += kernel[k + 2] * temp[(y * dstWidth + sx) * channels + c];
                }
                temp2[(y * dstWidth + x) * channels + c] = sum;
            }
        }
    }
    
    // Vertical pass
    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int k = -2; k <= 2; ++k) {
                    int sy = std::max(0, std::min(dstHeight - 1, y + k));
                    sum += kernel[k + 2] * temp2[(sy * dstWidth + x) * channels + c];
                }
                dst[(y * dstWidth + x) * channels + c] = sum;
            }
        }
    }
    
    return dst;
}

/**
 * @brief Build Gaussian pyramid
 */
inline std::vector<std::vector<float>> buildPyramid(
    const float* src, int srcWidth, int srcHeight, int channels,
    int maxLevel,
    std::vector<std::pair<int, int>>& sizes
) {
    std::vector<std::vector<float>> pyramid;
    sizes.clear();
    
    // Level 0 is the original image
    std::vector<float> level0(src, src + srcWidth * srcHeight * channels);
    pyramid.push_back(level0);
    sizes.push_back({srcWidth, srcHeight});
    
    int w = srcWidth, h = srcHeight;
    const float* current = src;
    
    for (int level = 1; level <= maxLevel && w > 1 && h > 1; ++level) {
        int newW, newH;
        auto levelData = pyrDown(
            pyramid.back().data(), w, h, channels,
            newW, newH
        );
        pyramid.push_back(levelData);
        sizes.push_back({newW, newH});
        w = newW;
        h = newH;
    }
    
    return pyramid;
}

/**
 * @brief Build Laplacian pyramid
 */
inline std::vector<std::vector<float>> buildLaplacianPyramid(
    const float* src, int srcWidth, int srcHeight, int channels,
    int maxLevel,
    std::vector<std::pair<int, int>>& sizes
) {
    // Build Gaussian pyramid first
    auto gaussian = buildPyramid(src, srcWidth, srcHeight, channels, maxLevel, sizes);
    
    std::vector<std::vector<float>> laplacian;
    
    // Laplacian[i] = Gaussian[i] - pyrUp(Gaussian[i+1])
    for (size_t i = 0; i < gaussian.size() - 1; ++i) {
        int w = sizes[i].first;
        int h = sizes[i].second;
        int upW, upH;
        
        auto expanded = pyrUp(
            gaussian[i + 1].data(),
            sizes[i + 1].first, sizes[i + 1].second, channels,
            upW, upH
        );
        
        // Adjust size if needed
        if (upW > w) upW = w;
        if (upH > h) upH = h;
        
        std::vector<float> lapLevel(w * h * channels);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < channels; ++c) {
                    int idx = (y * w + x) * channels + c;
                    float upVal = 0.0f;
                    if (x < upW && y < upH) {
                        upVal = expanded[(y * upW + x) * channels + c];
                    }
                    lapLevel[idx] = gaussian[i][idx] - upVal;
                }
            }
        }
        laplacian.push_back(lapLevel);
    }
    
    // Last level is the Gaussian (residual)
    laplacian.push_back(gaussian.back());
    
    return laplacian;
}

/**
 * @brief Reconstruct image from Laplacian pyramid
 */
inline std::vector<float> reconstructFromLaplacian(
    const std::vector<std::vector<float>>& laplacian,
    const std::vector<std::pair<int, int>>& sizes,
    int channels
) {
    if (laplacian.empty()) return {};
    
    // Start from the smallest level (residual)
    std::vector<float> current = laplacian.back();
    int w = sizes.back().first;
    int h = sizes.back().second;
    
    // Reconstruct from bottom to top
    for (int level = static_cast<int>(laplacian.size()) - 2; level >= 0; --level) {
        int upW, upH;
        auto expanded = pyrUp(current.data(), w, h, channels, upW, upH);
        
        w = sizes[level].first;
        h = sizes[level].second;
        
        // Add Laplacian
        current.resize(w * h * channels);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < channels; ++c) {
                    int idx = (y * w + x) * channels + c;
                    float upVal = 0.0f;
                    if (x < upW && y < upH) {
                        upVal = expanded[(y * upW + x) * channels + c];
                    }
                    current[idx] = laplacian[level][idx] + upVal;
                }
            }
        }
    }
    
    return current;
}

/**
 * @brief Pyramid blending of two images
 */
inline std::vector<float> pyramidBlend(
    const float* img1, const float* img2, const float* mask,
    int width, int height, int channels,
    int levels = 6
) {
    std::vector<std::pair<int, int>> sizes1, sizes2, sizesM;
    
    // Build Laplacian pyramids for both images
    auto lap1 = buildLaplacianPyramid(img1, width, height, channels, levels, sizes1);
    auto lap2 = buildLaplacianPyramid(img2, width, height, channels, levels, sizes2);
    
    // Build Gaussian pyramid for mask (1 channel)
    auto gaussMask = buildPyramid(mask, width, height, 1, levels, sizesM);
    
    // Blend at each level
    std::vector<std::vector<float>> blended;
    for (size_t level = 0; level < lap1.size(); ++level) {
        int w = sizes1[level].first;
        int h = sizes1[level].second;
        
        std::vector<float> blendLevel(w * h * channels);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float m = gaussMask[level][y * w + x];
                for (int c = 0; c < channels; ++c) {
                    int idx = (y * w + x) * channels + c;
                    blendLevel[idx] = m * lap1[level][idx] + (1.0f - m) * lap2[level][idx];
                }
            }
        }
        blended.push_back(blendLevel);
    }
    
    // Reconstruct
    return reconstructFromLaplacian(blended, sizes1, channels);
}

/**
 * @brief Scale space computation (for SIFT-like features)
 */
inline std::vector<std::vector<float>> buildScaleSpace(
    const float* src, int width, int height,
    int numOctaves,
    int numScales,
    double sigma0 = 1.6,
    std::vector<std::pair<int, int>>* octaveSizes = nullptr
) {
    std::vector<std::vector<float>> scaleSpace;
    
    if (octaveSizes) {
        octaveSizes->clear();
    }
    
    // Work with grayscale image
    std::vector<float> current(src, src + width * height);
    int w = width, h = height;
    
    for (int octave = 0; octave < numOctaves && w > 1 && h > 1; ++octave) {
        if (octaveSizes) {
            octaveSizes->push_back({w, h});
        }
        
        // Compute scales within octave
        for (int scale = 0; scale < numScales + 3; ++scale) {
            double sigma = sigma0 * std::pow(2.0, scale / static_cast<double>(numScales));
            int ksize = static_cast<int>(std::ceil(sigma * 6)) | 1;  // Ensure odd
            ksize = std::max(3, ksize);
            
            // Apply Gaussian blur
            std::vector<float> blurred(w * h);
            
            // Generate Gaussian kernel
            std::vector<float> kernel(ksize);
            float sum = 0.0f;
            int half = ksize / 2;
            for (int i = 0; i < ksize; ++i) {
                float x = static_cast<float>(i - half);
                kernel[i] = std::exp(-x * x / (2.0f * sigma * sigma));
                sum += kernel[i];
            }
            for (int i = 0; i < ksize; ++i) kernel[i] /= sum;
            
            // Separable convolution
            std::vector<float> temp(w * h);
            
            // Horizontal
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    float val = 0.0f;
                    for (int k = 0; k < ksize; ++k) {
                        int sx = std::max(0, std::min(w - 1, x + k - half));
                        val += kernel[k] * current[y * w + sx];
                    }
                    temp[y * w + x] = val;
                }
            }
            
            // Vertical
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    float val = 0.0f;
                    for (int k = 0; k < ksize; ++k) {
                        int sy = std::max(0, std::min(h - 1, y + k - half));
                        val += kernel[k] * temp[sy * w + x];
                    }
                    blurred[y * w + x] = val;
                }
            }
            
            scaleSpace.push_back(blurred);
        }
        
        // Downsample for next octave
        if (octave < numOctaves - 1) {
            // Use the scale at numScales index as base for next octave
            int newW = (w + 1) / 2;
            int newH = (h + 1) / 2;
            std::vector<float> downsampled(newW * newH);
            
            for (int y = 0; y < newH; ++y) {
                for (int x = 0; x < newW; ++x) {
                    downsampled[y * newW + x] = 
                        scaleSpace.back()[(y * 2) * w + (x * 2)];
                }
            }
            
            current = downsampled;
            w = newW;
            h = newH;
        }
    }
    
    return scaleSpace;
}

/**
 * @brief Compute Difference of Gaussians pyramid
 */
inline std::vector<std::vector<float>> buildDoGPyramid(
    const std::vector<std::vector<float>>& scaleSpace,
    int numOctaves,
    int numScales
) {
    std::vector<std::vector<float>> dogPyramid;
    
    int scalesPerOctave = numScales + 3;
    
    for (int octave = 0; octave < numOctaves; ++octave) {
        int baseIdx = octave * scalesPerOctave;
        
        for (int scale = 0; scale < scalesPerOctave - 1; ++scale) {
            const auto& img1 = scaleSpace[baseIdx + scale];
            const auto& img2 = scaleSpace[baseIdx + scale + 1];
            
            std::vector<float> dog(img1.size());
            for (size_t i = 0; i < img1.size(); ++i) {
                dog[i] = img2[i] - img1[i];
            }
            dogPyramid.push_back(dog);
        }
    }
    
    return dogPyramid;
}

} // namespace imgproc
} // namespace neurova

#endif // NEUROVA_IMGPROC_PYRAMID_HPP
