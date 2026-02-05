// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file histogram.hpp
 * @brief Histogram operations
 */

#ifndef NEUROVA_IMGPROC_HISTOGRAM_HPP
#define NEUROVA_IMGPROC_HISTOGRAM_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace neurova {
namespace imgproc {

/**
 * @brief Calculate histogram of grayscale image
 * @param input Input grayscale image
 * @param width Image width
 * @param height Image height
 * @param histSize Number of bins
 * @param minVal Minimum value for range
 * @param maxVal Maximum value for range
 * @return Histogram as vector
 */
inline std::vector<float> calcHist(
    const float* input, int width, int height,
    int histSize = 256,
    float minVal = 0.0f,
    float maxVal = 256.0f
) {
    std::vector<float> hist(histSize, 0.0f);
    float range = maxVal - minVal;
    
    for (int i = 0; i < width * height; ++i) {
        float value = input[i];
        int bin = static_cast<int>((value - minVal) / range * histSize);
        bin = std::clamp(bin, 0, histSize - 1);
        hist[bin] += 1.0f;
    }
    
    return hist;
}

/**
 * @brief Normalize histogram
 * @param hist Input histogram
 * @param normType Normalization type (0=L1, 1=L2, 2=MinMax)
 * @param alpha Target max for MinMax or norm value
 * @param beta Target min for MinMax
 * @return Normalized histogram
 */
inline std::vector<float> normalizeHist(
    const std::vector<float>& hist,
    int normType = 2,
    float alpha = 1.0f,
    float beta = 0.0f
) {
    std::vector<float> result(hist.size());
    
    if (normType == 0) {
        // L1 normalization
        float sum = 0.0f;
        for (auto v : hist) sum += std::abs(v);
        if (sum > 0) {
            for (size_t i = 0; i < hist.size(); ++i) {
                result[i] = hist[i] / sum * alpha;
            }
        }
    } else if (normType == 1) {
        // L2 normalization
        float sum = 0.0f;
        for (auto v : hist) sum += v * v;
        sum = std::sqrt(sum);
        if (sum > 0) {
            for (size_t i = 0; i < hist.size(); ++i) {
                result[i] = hist[i] / sum * alpha;
            }
        }
    } else {
        // MinMax normalization
        float minV = *std::min_element(hist.begin(), hist.end());
        float maxV = *std::max_element(hist.begin(), hist.end());
        float range = maxV - minV;
        
        if (range > 0) {
            for (size_t i = 0; i < hist.size(); ++i) {
                result[i] = (hist[i] - minV) / range * (alpha - beta) + beta;
            }
        }
    }
    
    return result;
}

/**
 * @brief Equalize histogram of grayscale image
 * @param input Input grayscale image
 * @param width Image width
 * @param height Image height
 * @return Histogram-equalized image
 */
inline std::vector<float> equalizeHist(
    const float* input, int width, int height
) {
    // Calculate histogram
    auto hist = calcHist(input, width, height, 256, 0.0f, 256.0f);
    
    // Calculate CDF
    std::vector<float> cdf(256);
    cdf[0] = hist[0];
    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + hist[i];
    }
    
    // Normalize CDF
    float total = static_cast<float>(width * height);
    float cdfMin = 0.0f;
    for (auto v : cdf) {
        if (v > 0) {
            cdfMin = v;
            break;
        }
    }
    
    std::vector<float> lut(256);
    for (int i = 0; i < 256; ++i) {
        lut[i] = std::round((cdf[i] - cdfMin) / (total - cdfMin) * 255.0f);
    }
    
    // Apply LUT
    std::vector<float> output(width * height);
    for (int i = 0; i < width * height; ++i) {
        int idx = std::clamp(static_cast<int>(input[i]), 0, 255);
        output[i] = lut[idx];
    }
    
    return output;
}

/**
 * @brief Compare two histograms
 * @param h1 First histogram
 * @param h2 Second histogram
 * @param method Comparison method (0=Correlation, 1=Chi-Square, 2=Intersection, 3=Bhattacharyya)
 * @return Similarity/distance value
 */
inline float compareHist(
    const std::vector<float>& h1,
    const std::vector<float>& h2,
    int method = 0
) {
    if (h1.size() != h2.size()) return 0.0f;
    
    int n = static_cast<int>(h1.size());
    
    if (method == 0) {
        // Correlation
        float mean1 = 0.0f, mean2 = 0.0f;
        for (auto v : h1) mean1 += v;
        for (auto v : h2) mean2 += v;
        mean1 /= n;
        mean2 /= n;
        
        float num = 0.0f, den1 = 0.0f, den2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            float a = h1[i] - mean1;
            float b = h2[i] - mean2;
            num += a * b;
            den1 += a * a;
            den2 += b * b;
        }
        
        return num / (std::sqrt(den1 * den2) + 1e-10f);
    } else if (method == 1) {
        // Chi-Square
        float result = 0.0f;
        for (int i = 0; i < n; ++i) {
            float den = h1[i] + h2[i];
            if (den > 0) {
                float diff = h1[i] - h2[i];
                result += diff * diff / den;
            }
        }
        return result;
    } else if (method == 2) {
        // Intersection
        float result = 0.0f;
        for (int i = 0; i < n; ++i) {
            result += std::min(h1[i], h2[i]);
        }
        return result;
    } else {
        // Bhattacharyya
        float sum1 = 0.0f, sum2 = 0.0f;
        for (auto v : h1) sum1 += v;
        for (auto v : h2) sum2 += v;
        
        float bc = 0.0f;
        for (int i = 0; i < n; ++i) {
            bc += std::sqrt(h1[i] * h2[i]);
        }
        bc /= std::sqrt(sum1 * sum2);
        
        return std::sqrt(1.0f - bc);
    }
}

/**
 * @brief Apply look-up table to image
 */
inline std::vector<float> LUT(
    const float* input, int width, int height,
    const float* lut, int lutSize = 256
) {
    std::vector<float> output(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        int idx = std::clamp(static_cast<int>(input[i]), 0, lutSize - 1);
        output[i] = lut[idx];
    }
    
    return output;
}

/**
 * @brief CLAHE (Contrast Limited Adaptive Histogram Equalization)
 */
inline std::vector<float> CLAHE(
    const float* input, int width, int height,
    float clipLimit = 40.0f,
    int tileGridSizeX = 8,
    int tileGridSizeY = 8
) {
    std::vector<float> output(width * height);
    
    int tileWidth = width / tileGridSizeX;
    int tileHeight = height / tileGridSizeY;
    
    // Process each tile
    std::vector<std::vector<float>> tileLUTs(tileGridSizeX * tileGridSizeY);
    
    for (int ty = 0; ty < tileGridSizeY; ++ty) {
        for (int tx = 0; tx < tileGridSizeX; ++tx) {
            int startX = tx * tileWidth;
            int startY = ty * tileHeight;
            int endX = std::min((tx + 1) * tileWidth, width);
            int endY = std::min((ty + 1) * tileHeight, height);
            
            // Calculate histogram for tile
            std::vector<float> hist(256, 0.0f);
            int pixelCount = 0;
            
            for (int y = startY; y < endY; ++y) {
                for (int x = startX; x < endX; ++x) {
                    int bin = std::clamp(static_cast<int>(input[y * width + x]), 0, 255);
                    hist[bin] += 1.0f;
                    ++pixelCount;
                }
            }
            
            // Clip histogram
            float clip = clipLimit * pixelCount / 256.0f;
            float excess = 0.0f;
            
            for (int i = 0; i < 256; ++i) {
                if (hist[i] > clip) {
                    excess += hist[i] - clip;
                    hist[i] = clip;
                }
            }
            
            // Redistribute excess
            float avgExcess = excess / 256.0f;
            for (int i = 0; i < 256; ++i) {
                hist[i] += avgExcess;
            }
            
            // Compute CDF and LUT
            std::vector<float>& lut = tileLUTs[ty * tileGridSizeX + tx];
            lut.resize(256);
            
            float cdf = 0.0f;
            for (int i = 0; i < 256; ++i) {
                cdf += hist[i];
                lut[i] = cdf / pixelCount * 255.0f;
            }
        }
    }
    
    // Bilinear interpolation between tiles
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float fx = static_cast<float>(x) / tileWidth - 0.5f;
            float fy = static_cast<float>(y) / tileHeight - 0.5f;
            
            int tx1 = std::clamp(static_cast<int>(std::floor(fx)), 0, tileGridSizeX - 1);
            int ty1 = std::clamp(static_cast<int>(std::floor(fy)), 0, tileGridSizeY - 1);
            int tx2 = std::min(tx1 + 1, tileGridSizeX - 1);
            int ty2 = std::min(ty1 + 1, tileGridSizeY - 1);
            
            float dx = fx - tx1;
            float dy = fy - ty1;
            dx = std::clamp(dx, 0.0f, 1.0f);
            dy = std::clamp(dy, 0.0f, 1.0f);
            
            int bin = std::clamp(static_cast<int>(input[y * width + x]), 0, 255);
            
            float v11 = tileLUTs[ty1 * tileGridSizeX + tx1][bin];
            float v12 = tileLUTs[ty1 * tileGridSizeX + tx2][bin];
            float v21 = tileLUTs[ty2 * tileGridSizeX + tx1][bin];
            float v22 = tileLUTs[ty2 * tileGridSizeX + tx2][bin];
            
            output[y * width + x] = (1 - dx) * (1 - dy) * v11 +
                                    dx * (1 - dy) * v12 +
                                    (1 - dx) * dy * v21 +
                                    dx * dy * v22;
        }
    }
    
    return output;
}

/**
 * @brief Calculate back projection
 */
inline std::vector<float> calcBackProject(
    const float* input, int width, int height,
    const std::vector<float>& hist,
    float minVal = 0.0f,
    float maxVal = 256.0f
) {
    std::vector<float> output(width * height);
    int histSize = static_cast<int>(hist.size());
    float range = maxVal - minVal;
    
    for (int i = 0; i < width * height; ++i) {
        float value = input[i];
        int bin = static_cast<int>((value - minVal) / range * histSize);
        bin = std::clamp(bin, 0, histSize - 1);
        output[i] = hist[bin];
    }
    
    return output;
}

/**
 * @brief Adjust brightness and contrast
 * output = alpha * input + beta
 */
inline std::vector<float> adjustBrightnessContrast(
    const float* input, int width, int height,
    float alpha = 1.0f,  // Contrast
    float beta = 0.0f    // Brightness
) {
    std::vector<float> output(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        output[i] = std::clamp(alpha * input[i] + beta, 0.0f, 255.0f);
    }
    
    return output;
}

/**
 * @brief Gamma correction
 */
inline std::vector<float> gammaCorrection(
    const float* input, int width, int height,
    float gamma
) {
    // Build LUT
    std::vector<float> lut(256);
    float invGamma = 1.0f / gamma;
    
    for (int i = 0; i < 256; ++i) {
        lut[i] = std::pow(i / 255.0f, invGamma) * 255.0f;
    }
    
    return LUT(input, width, height, lut.data());
}

/**
 * @brief Auto contrast adjustment
 */
inline std::vector<float> autoContrast(
    const float* input, int width, int height,
    float lowPercentile = 0.5f,
    float highPercentile = 99.5f
) {
    // Sort values to find percentiles
    std::vector<float> sorted(input, input + width * height);
    std::sort(sorted.begin(), sorted.end());
    
    int lowIdx = static_cast<int>(sorted.size() * lowPercentile / 100.0f);
    int highIdx = static_cast<int>(sorted.size() * highPercentile / 100.0f);
    highIdx = std::min(highIdx, static_cast<int>(sorted.size()) - 1);
    
    float low = sorted[lowIdx];
    float high = sorted[highIdx];
    
    if (high <= low) {
        return std::vector<float>(input, input + width * height);
    }
    
    std::vector<float> output(width * height);
    float scale = 255.0f / (high - low);
    
    for (int i = 0; i < width * height; ++i) {
        output[i] = std::clamp((input[i] - low) * scale, 0.0f, 255.0f);
    }
    
    return output;
}

} // namespace imgproc
} // namespace neurova

#endif // NEUROVA_IMGPROC_HISTOGRAM_HPP
