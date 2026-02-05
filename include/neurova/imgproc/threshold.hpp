// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file threshold.hpp
 * @brief Image thresholding operations
 */

#ifndef NEUROVA_IMGPROC_THRESHOLD_HPP
#define NEUROVA_IMGPROC_THRESHOLD_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace neurova {
namespace imgproc {

// Threshold types
constexpr int THRESH_BINARY = 0;
constexpr int THRESH_BINARY_INV = 1;
constexpr int THRESH_TRUNC = 2;
constexpr int THRESH_TOZERO = 3;
constexpr int THRESH_TOZERO_INV = 4;
constexpr int THRESH_OTSU = 8;
constexpr int THRESH_TRIANGLE = 16;

// Adaptive threshold methods
constexpr int ADAPTIVE_THRESH_MEAN_C = 0;
constexpr int ADAPTIVE_THRESH_GAUSSIAN_C = 1;

/**
 * @brief Simple thresholding
 * @param input Input grayscale image
 * @param width Image width
 * @param height Image height
 * @param thresh Threshold value
 * @param maxval Maximum value for BINARY thresholding
 * @param type Threshold type
 * @return Thresholded image and computed threshold (for OTSU/TRIANGLE)
 */
inline std::pair<std::vector<float>, float> threshold(
    const float* input, int width, int height,
    float thresh,
    float maxval,
    int type
) {
    std::vector<float> output(width * height);
    float computedThresh = thresh;
    
    // Handle automatic threshold computation
    if (type & THRESH_OTSU) {
        // Compute Otsu's threshold
        std::vector<int> histogram(256, 0);
        for (int i = 0; i < width * height; ++i) {
            int bin = std::clamp(static_cast<int>(input[i]), 0, 255);
            histogram[bin]++;
        }
        
        int total = width * height;
        float sum = 0.0f;
        for (int i = 0; i < 256; ++i) {
            sum += i * histogram[i];
        }
        
        float sumB = 0.0f;
        int wB = 0;
        float maxVariance = 0.0f;
        
        for (int t = 0; t < 256; ++t) {
            wB += histogram[t];
            if (wB == 0) continue;
            
            int wF = total - wB;
            if (wF == 0) break;
            
            sumB += t * histogram[t];
            float mB = sumB / wB;
            float mF = (sum - sumB) / wF;
            
            float variance = static_cast<float>(wB) * wF * (mB - mF) * (mB - mF);
            
            if (variance > maxVariance) {
                maxVariance = variance;
                computedThresh = static_cast<float>(t);
            }
        }
        
        type &= ~THRESH_OTSU;  // Remove flag
    } else if (type & THRESH_TRIANGLE) {
        // Compute triangle threshold
        std::vector<int> histogram(256, 0);
        for (int i = 0; i < width * height; ++i) {
            int bin = std::clamp(static_cast<int>(input[i]), 0, 255);
            histogram[bin]++;
        }
        
        // Find peak and endpoints
        int peak = 0, peakVal = 0;
        int left = 0, right = 255;
        
        for (int i = 0; i < 256; ++i) {
            if (histogram[i] > peakVal) {
                peakVal = histogram[i];
                peak = i;
            }
        }
        
        // Find first and last non-zero bins
        while (left < 256 && histogram[left] == 0) left++;
        while (right > 0 && histogram[right] == 0) right--;
        
        // Determine which side to use
        bool flipLR = (peak - left) < (right - peak);
        if (flipLR) {
            std::swap(left, right);
        }
        
        // Line from peak to endpoint
        float A = static_cast<float>(peakVal);
        float B = static_cast<float>(left - peak);
        float C = static_cast<float>(-peakVal * left);
        float norm = std::sqrt(A * A + B * B);
        
        float maxDist = 0.0f;
        for (int i = std::min(peak, left); i <= std::max(peak, left); ++i) {
            float dist = std::abs(A * i + B * histogram[i] + C) / norm;
            if (dist > maxDist) {
                maxDist = dist;
                computedThresh = static_cast<float>(i);
            }
        }
        
        type &= ~THRESH_TRIANGLE;
    }
    
    // Apply threshold
    for (int i = 0; i < width * height; ++i) {
        float value = input[i];
        
        switch (type) {
            case THRESH_BINARY:
                output[i] = (value > computedThresh) ? maxval : 0.0f;
                break;
            case THRESH_BINARY_INV:
                output[i] = (value > computedThresh) ? 0.0f : maxval;
                break;
            case THRESH_TRUNC:
                output[i] = (value > computedThresh) ? computedThresh : value;
                break;
            case THRESH_TOZERO:
                output[i] = (value > computedThresh) ? value : 0.0f;
                break;
            case THRESH_TOZERO_INV:
                output[i] = (value > computedThresh) ? 0.0f : value;
                break;
            default:
                output[i] = (value > computedThresh) ? maxval : 0.0f;
                break;
        }
    }
    
    return {output, computedThresh};
}

/**
 * @brief Adaptive thresholding
 * @param input Input grayscale image
 * @param width Image width
 * @param height Image height
 * @param maxval Maximum value for BINARY thresholding
 * @param adaptiveMethod Adaptive method (MEAN or GAUSSIAN)
 * @param thresholdType Threshold type (BINARY or BINARY_INV)
 * @param blockSize Size of neighborhood (must be odd)
 * @param C Constant subtracted from mean/weighted mean
 * @return Thresholded image
 */
inline std::vector<float> adaptiveThreshold(
    const float* input, int width, int height,
    float maxval,
    int adaptiveMethod,
    int thresholdType,
    int blockSize,
    float C
) {
    std::vector<float> output(width * height);
    
    if (blockSize % 2 == 0) blockSize += 1;
    int pad = blockSize / 2;
    
    // Compute local means
    std::vector<float> localMean(width * height);
    
    if (adaptiveMethod == ADAPTIVE_THRESH_MEAN_C) {
        // Box filter
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
                
                localMean[y * width + x] = sum / count;
            }
        }
    } else {
        // Gaussian weighted
        // Generate Gaussian kernel
        std::vector<float> kernel(blockSize * blockSize);
        float sigma = blockSize / 6.0f;
        float sigma2 = 2.0f * sigma * sigma;
        float sum = 0.0f;
        
        for (int ky = 0; ky < blockSize; ++ky) {
            for (int kx = 0; kx < blockSize; ++kx) {
                float dx = static_cast<float>(kx - pad);
                float dy = static_cast<float>(ky - pad);
                kernel[ky * blockSize + kx] = std::exp(-(dx * dx + dy * dy) / sigma2);
                sum += kernel[ky * blockSize + kx];
            }
        }
        for (auto& v : kernel) v /= sum;
        
        // Apply Gaussian filter
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float weighted = 0.0f;
                
                for (int ky = -pad; ky <= pad; ++ky) {
                    for (int kx = -pad; kx <= pad; ++kx) {
                        int nx = std::clamp(x + kx, 0, width - 1);
                        int ny = std::clamp(y + ky, 0, height - 1);
                        weighted += input[ny * width + nx] * 
                                   kernel[(ky + pad) * blockSize + (kx + pad)];
                    }
                }
                
                localMean[y * width + x] = weighted;
            }
        }
    }
    
    // Apply threshold
    for (int i = 0; i < width * height; ++i) {
        float thresh = localMean[i] - C;
        
        if (thresholdType == THRESH_BINARY) {
            output[i] = (input[i] > thresh) ? maxval : 0.0f;
        } else {
            output[i] = (input[i] > thresh) ? 0.0f : maxval;
        }
    }
    
    return output;
}

/**
 * @brief Check if values are in range
 * @param input Input image (any number of channels)
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels
 * @param lowerb Lower boundary
 * @param upperb Upper boundary
 * @return Binary mask
 */
inline std::vector<uint8_t> inRange(
    const float* input, int width, int height, int channels,
    const float* lowerb,
    const float* upperb
) {
    std::vector<uint8_t> output(width * height, 0);
    
    for (int i = 0; i < width * height; ++i) {
        bool inRange = true;
        
        for (int c = 0; c < channels; ++c) {
            float value = input[i * channels + c];
            if (value < lowerb[c] || value > upperb[c]) {
                inRange = false;
                break;
            }
        }
        
        output[i] = inRange ? 255 : 0;
    }
    
    return output;
}

/**
 * @brief Multi-level thresholding using multi-Otsu
 * @param input Input grayscale image
 * @param width Image width
 * @param height Image height
 * @param nClasses Number of classes (thresholds = nClasses - 1)
 * @return Vector of threshold values
 */
inline std::vector<float> multiOtsuThreshold(
    const float* input, int width, int height,
    int nClasses = 3
) {
    int nThresholds = nClasses - 1;
    
    // Compute histogram
    std::vector<int> histogram(256, 0);
    for (int i = 0; i < width * height; ++i) {
        int bin = std::clamp(static_cast<int>(input[i]), 0, 255);
        histogram[bin]++;
    }
    
    // Normalize histogram
    int total = width * height;
    std::vector<float> prob(256);
    for (int i = 0; i < 256; ++i) {
        prob[i] = static_cast<float>(histogram[i]) / total;
    }
    
    // Compute cumulative sums
    std::vector<float> P(256), S(256);
    P[0] = prob[0];
    S[0] = 0.0f;
    for (int i = 1; i < 256; ++i) {
        P[i] = P[i - 1] + prob[i];
        S[i] = S[i - 1] + i * prob[i];
    }
    
    std::vector<float> thresholds(nThresholds);
    
    if (nClasses == 2) {
        // Standard Otsu
        float maxVariance = 0.0f;
        for (int t = 0; t < 256; ++t) {
            if (P[t] > 0 && P[t] < 1) {
                float mu1 = S[t] / P[t];
                float mu2 = (S[255] - S[t]) / (1 - P[t]);
                float variance = P[t] * (1 - P[t]) * (mu1 - mu2) * (mu1 - mu2);
                
                if (variance > maxVariance) {
                    maxVariance = variance;
                    thresholds[0] = static_cast<float>(t);
                }
            }
        }
    } else if (nClasses == 3) {
        // Two thresholds
        float maxVariance = 0.0f;
        
        for (int t1 = 0; t1 < 254; ++t1) {
            for (int t2 = t1 + 1; t2 < 255; ++t2) {
                float w0 = P[t1];
                float w1 = P[t2] - P[t1];
                float w2 = 1 - P[t2];
                
                if (w0 > 0 && w1 > 0 && w2 > 0) {
                    float mu0 = S[t1] / w0;
                    float mu1 = (S[t2] - S[t1]) / w1;
                    float mu2 = (S[255] - S[t2]) / w2;
                    float muT = S[255];
                    
                    float variance = w0 * (mu0 - muT) * (mu0 - muT) +
                                    w1 * (mu1 - muT) * (mu1 - muT) +
                                    w2 * (mu2 - muT) * (mu2 - muT);
                    
                    if (variance > maxVariance) {
                        maxVariance = variance;
                        thresholds[0] = static_cast<float>(t1);
                        thresholds[1] = static_cast<float>(t2);
                    }
                }
            }
        }
    }
    
    return thresholds;
}

/**
 * @brief Apply thresholds to segment image
 */
inline std::vector<uint8_t> applyMultiThreshold(
    const float* input, int width, int height,
    const std::vector<float>& thresholds
) {
    std::vector<uint8_t> output(width * height);
    int nClasses = static_cast<int>(thresholds.size()) + 1;
    
    for (int i = 0; i < width * height; ++i) {
        int label = 0;
        for (size_t t = 0; t < thresholds.size(); ++t) {
            if (input[i] > thresholds[t]) {
                label = static_cast<int>(t) + 1;
            }
        }
        output[i] = static_cast<uint8_t>(label * 255 / (nClasses - 1));
    }
    
    return output;
}

/**
 * @brief Sauvola local thresholding
 */
inline std::vector<float> sauvolaThreshold(
    const float* input, int width, int height,
    int windowSize = 15,
    float k = 0.5f,
    float R = 128.0f
) {
    std::vector<float> output(width * height);
    
    int pad = windowSize / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            float sumSq = 0.0f;
            int count = 0;
            
            for (int ky = -pad; ky <= pad; ++ky) {
                for (int kx = -pad; kx <= pad; ++kx) {
                    int nx = std::clamp(x + kx, 0, width - 1);
                    int ny = std::clamp(y + ky, 0, height - 1);
                    float val = input[ny * width + nx];
                    sum += val;
                    sumSq += val * val;
                    ++count;
                }
            }
            
            float mean = sum / count;
            float variance = sumSq / count - mean * mean;
            float stddev = std::sqrt(std::max(0.0f, variance));
            
            float thresh = mean * (1.0f + k * (stddev / R - 1.0f));
            output[y * width + x] = (input[y * width + x] > thresh) ? 255.0f : 0.0f;
        }
    }
    
    return output;
}

/**
 * @brief Niblack local thresholding
 */
inline std::vector<float> niblackThreshold(
    const float* input, int width, int height,
    int windowSize = 15,
    float k = -0.2f
) {
    std::vector<float> output(width * height);
    
    int pad = windowSize / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            float sumSq = 0.0f;
            int count = 0;
            
            for (int ky = -pad; ky <= pad; ++ky) {
                for (int kx = -pad; kx <= pad; ++kx) {
                    int nx = std::clamp(x + kx, 0, width - 1);
                    int ny = std::clamp(y + ky, 0, height - 1);
                    float val = input[ny * width + nx];
                    sum += val;
                    sumSq += val * val;
                    ++count;
                }
            }
            
            float mean = sum / count;
            float variance = sumSq / count - mean * mean;
            float stddev = std::sqrt(std::max(0.0f, variance));
            
            float thresh = mean + k * stddev;
            output[y * width + x] = (input[y * width + x] > thresh) ? 255.0f : 0.0f;
        }
    }
    
    return output;
}

} // namespace imgproc
} // namespace neurova

#endif // NEUROVA_IMGPROC_THRESHOLD_HPP
