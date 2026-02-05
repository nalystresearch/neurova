// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file video/background.hpp
 * @brief Background subtraction algorithms
 */

#ifndef NEUROVA_VIDEO_BACKGROUND_HPP
#define NEUROVA_VIDEO_BACKGROUND_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace neurova {
namespace video {

// ============================================================================
// Background Subtractor Base
// ============================================================================

class BackgroundSubtractorBase {
public:
    virtual ~BackgroundSubtractorBase() = default;
    
    virtual void apply(const uint8_t* image, int w, int h, int channels,
                      uint8_t* fgmask, float learningRate = -1) = 0;
    
    virtual void getBackgroundImage(uint8_t* background) const = 0;
};

// ============================================================================
// MOG2 Background Subtractor
// ============================================================================

class BackgroundSubtractorMOG2 : public BackgroundSubtractorBase {
public:
    BackgroundSubtractorMOG2(int history = 500, float varThreshold = 16.0f, 
                             bool detectShadows = true)
        : history_(history), varThreshold_(varThreshold), detectShadows_(detectShadows),
          nMixtures_(5), backgroundRatio_(0.9f), varMin_(4.0f), 
          varMax_(5.0f * varThreshold), varInit_(15.0f),
          shadowValue_(127), shadowThreshold_(0.5f),
          initialized_(false), frameCount_(0), width_(0), height_(0), channels_(0) {}
    
    void apply(const uint8_t* image, int w, int h, int channels,
               uint8_t* fgmask, float learningRate = -1) override {
        
        if (!initialized_ || w != width_ || h != height_) {
            initialize(w, h, channels);
        }
        
        frameCount_++;
        
        float alpha = (learningRate < 0) ? 
            1.0f / std::min(frameCount_, history_) : learningRate;
        
        int pixels = w * h;
        
        for (int p = 0; p < pixels; ++p) {
            int x = p % w;
            int y = p / w;
            
            // Get pixel values
            std::vector<float> pixelVal(channels);
            for (int c = 0; c < channels; ++c) {
                pixelVal[c] = static_cast<float>(image[p * channels + c]);
            }
            
            bool matched = false;
            int matchedK = -1;
            
            // Find matching component
            for (int k = 0; k < nMixtures_; ++k) {
                float distSq = 0;
                for (int c = 0; c < channels; ++c) {
                    float diff = pixelVal[c] - means_[p * nMixtures_ * channels + k * channels + c];
                    distSq += diff * diff;
                }
                distSq /= (variances_[p * nMixtures_ + k] + 1e-6f);
                
                if (distSq < varThreshold_) {
                    matched = true;
                    matchedK = k;
                    break;
                }
            }
            
            if (matched) {
                // Update matched component
                int k = matchedK;
                weights_[p * nMixtures_ + k] += alpha * (1 - weights_[p * nMixtures_ + k]);
                
                float rho = alpha / (weights_[p * nMixtures_ + k] + 1e-6f);
                rho = std::clamp(rho, 0.0f, 1.0f);
                
                float newVar = 0;
                for (int c = 0; c < channels; ++c) {
                    float& mean = means_[p * nMixtures_ * channels + k * channels + c];
                    float diff = pixelVal[c] - mean;
                    mean = (1 - rho) * mean + rho * pixelVal[c];
                    newVar += diff * diff;
                }
                
                variances_[p * nMixtures_ + k] = (1 - rho) * variances_[p * nMixtures_ + k] + rho * newVar;
                variances_[p * nMixtures_ + k] = std::clamp(variances_[p * nMixtures_ + k], varMin_, varMax_);
                
                // Decay other weights
                for (int j = 0; j < nMixtures_; ++j) {
                    if (j != k) {
                        weights_[p * nMixtures_ + j] *= (1 - alpha);
                    }
                }
            } else {
                // Replace weakest component
                int weakest = 0;
                float minWeight = weights_[p * nMixtures_];
                for (int k = 1; k < nMixtures_; ++k) {
                    if (weights_[p * nMixtures_ + k] < minWeight) {
                        minWeight = weights_[p * nMixtures_ + k];
                        weakest = k;
                    }
                }
                
                weights_[p * nMixtures_ + weakest] = alpha;
                for (int c = 0; c < channels; ++c) {
                    means_[p * nMixtures_ * channels + weakest * channels + c] = pixelVal[c];
                }
                variances_[p * nMixtures_ + weakest] = varInit_;
            }
            
            // Normalize weights
            float sumW = 0;
            for (int k = 0; k < nMixtures_; ++k) {
                sumW += weights_[p * nMixtures_ + k];
            }
            for (int k = 0; k < nMixtures_; ++k) {
                weights_[p * nMixtures_ + k] /= (sumW + 1e-6f);
            }
            
            // Determine foreground/background
            // Check if matches dominant component
            float distSq0 = 0;
            for (int c = 0; c < channels; ++c) {
                float diff = pixelVal[c] - means_[p * nMixtures_ * channels + c];
                distSq0 += diff * diff;
            }
            distSq0 /= (variances_[p * nMixtures_] + 1e-6f);
            
            bool isBg = (distSq0 < varThreshold_);
            
            if (!isBg) {
                fgmask[p] = 255;
                
                // Shadow detection
                if (detectShadows_) {
                    float bgIntensity = 0, imgIntensity = 0;
                    for (int c = 0; c < channels; ++c) {
                        bgIntensity += means_[p * nMixtures_ * channels + c];
                        imgIntensity += pixelVal[c];
                    }
                    bgIntensity /= channels;
                    imgIntensity /= channels;
                    
                    float ratio = imgIntensity / (bgIntensity + 1e-6f);
                    if (ratio > shadowThreshold_ && ratio < 1.0f) {
                        fgmask[p] = shadowValue_;
                    }
                }
            } else {
                fgmask[p] = 0;
            }
            
            // Update background
            for (int c = 0; c < channels; ++c) {
                background_[p * channels + c] = static_cast<uint8_t>(
                    std::clamp(means_[p * nMixtures_ * channels + c], 0.0f, 255.0f));
            }
        }
    }
    
    void getBackgroundImage(uint8_t* background) const override {
        if (!initialized_) return;
        std::copy(background_.begin(), background_.end(), background);
    }
    
    // Getters/Setters
    int getHistory() const { return history_; }
    void setHistory(int h) { history_ = h; }
    
    int getNMixtures() const { return nMixtures_; }
    void setNMixtures(int n) { nMixtures_ = n; }
    
    float getBackgroundRatio() const { return backgroundRatio_; }
    void setBackgroundRatio(float r) { backgroundRatio_ = r; }
    
    float getVarThreshold() const { return varThreshold_; }
    void setVarThreshold(float t) { varThreshold_ = t; }
    
    bool getDetectShadows() const { return detectShadows_; }
    void setDetectShadows(bool d) { detectShadows_ = d; }
    
    int getShadowValue() const { return shadowValue_; }
    void setShadowValue(int v) { shadowValue_ = v; }
    
    float getShadowThreshold() const { return shadowThreshold_; }
    void setShadowThreshold(float t) { shadowThreshold_ = t; }

private:
    void initialize(int w, int h, int c) {
        width_ = w;
        height_ = h;
        channels_ = c;
        int pixels = w * h;
        
        weights_.resize(pixels * nMixtures_, 0.0f);
        means_.resize(pixels * nMixtures_ * c, 0.0f);
        variances_.resize(pixels * nMixtures_, varInit_);
        background_.resize(pixels * c, 0);
        
        // Initialize first component with 1.0 weight
        for (int p = 0; p < pixels; ++p) {
            weights_[p * nMixtures_] = 1.0f;
        }
        
        initialized_ = true;
        frameCount_ = 0;
    }
    
    int history_;
    float varThreshold_;
    bool detectShadows_;
    int nMixtures_;
    float backgroundRatio_;
    float varMin_, varMax_, varInit_;
    int shadowValue_;
    float shadowThreshold_;
    
    bool initialized_;
    int frameCount_;
    int width_, height_, channels_;
    
    std::vector<float> weights_;
    std::vector<float> means_;
    std::vector<float> variances_;
    std::vector<uint8_t> background_;
};

// ============================================================================
// KNN Background Subtractor
// ============================================================================

class BackgroundSubtractorKNN : public BackgroundSubtractorBase {
public:
    BackgroundSubtractorKNN(int history = 500, float dist2Threshold = 400.0f,
                           bool detectShadows = true)
        : history_(history), dist2Threshold_(dist2Threshold), detectShadows_(detectShadows),
          knnSamples_(7), nSamples_(10), shadowValue_(127), shadowThreshold_(0.5f),
          initialized_(false), frameCount_(0), width_(0), height_(0), channels_(0) {}
    
    void apply(const uint8_t* image, int w, int h, int channels,
               uint8_t* fgmask, float learningRate = -1) override {
        
        if (!initialized_ || w != width_ || h != height_) {
            initialize(w, h, channels);
        }
        
        frameCount_++;
        
        float alpha = (learningRate < 0) ?
            1.0f / std::min(frameCount_, history_) : learningRate;
        
        int pixels = w * h;
        
        for (int p = 0; p < pixels; ++p) {
            // Get pixel values
            std::vector<float> pixelVal(channels);
            for (int c = 0; c < channels; ++c) {
                pixelVal[c] = static_cast<float>(image[p * channels + c]);
            }
            
            // Count neighbors within threshold
            int neighbors = 0;
            for (int s = 0; s < nSamples_; ++s) {
                float distSq = 0;
                for (int c = 0; c < channels; ++c) {
                    float diff = pixelVal[c] - samples_[p * nSamples_ * channels + s * channels + c];
                    distSq += diff * diff;
                }
                if (distSq < dist2Threshold_) {
                    neighbors++;
                }
            }
            
            // Foreground if not enough neighbors
            bool isFg = (neighbors < knnSamples_);
            
            if (isFg) {
                fgmask[p] = 255;
                
                // Shadow detection
                if (detectShadows_) {
                    float bgIntensity = 0, imgIntensity = 0;
                    for (int c = 0; c < channels; ++c) {
                        bgIntensity += background_[p * channels + c];
                        imgIntensity += pixelVal[c];
                    }
                    bgIntensity /= channels;
                    imgIntensity /= channels;
                    
                    float ratio = imgIntensity / (bgIntensity + 1e-6f);
                    if (ratio > shadowThreshold_ && ratio < 1.0f) {
                        fgmask[p] = shadowValue_;
                    }
                }
            } else {
                fgmask[p] = 0;
                
                // Update random sample with probability
                float rand01 = static_cast<float>(rand()) / RAND_MAX;
                if (rand01 < alpha) {
                    int sampleIdx = rand() % nSamples_;
                    for (int c = 0; c < channels; ++c) {
                        samples_[p * nSamples_ * channels + sampleIdx * channels + c] = pixelVal[c];
                    }
                }
            }
            
            // Update background as median (simplified: use first sample)
            for (int c = 0; c < channels; ++c) {
                background_[p * channels + c] = static_cast<uint8_t>(
                    std::clamp(samples_[p * nSamples_ * channels + c], 0.0f, 255.0f));
            }
        }
    }
    
    void getBackgroundImage(uint8_t* background) const override {
        if (!initialized_) return;
        std::copy(background_.begin(), background_.end(), background);
    }
    
    // Getters/Setters
    int getHistory() const { return history_; }
    void setHistory(int h) { history_ = h; }
    
    float getDist2Threshold() const { return dist2Threshold_; }
    void setDist2Threshold(float t) { dist2Threshold_ = t; }
    
    bool getDetectShadows() const { return detectShadows_; }
    void setDetectShadows(bool d) { detectShadows_ = d; }
    
    int getShadowValue() const { return shadowValue_; }
    void setShadowValue(int v) { shadowValue_ = v; }
    
    int getkNNSamples() const { return knnSamples_; }
    void setkNNSamples(int n) { knnSamples_ = n; }

private:
    void initialize(int w, int h, int c) {
        width_ = w;
        height_ = h;
        channels_ = c;
        int pixels = w * h;
        
        samples_.resize(pixels * nSamples_ * c, 0.0f);
        background_.resize(pixels * c, 0);
        
        initialized_ = true;
        frameCount_ = 0;
    }
    
    int history_;
    float dist2Threshold_;
    bool detectShadows_;
    int knnSamples_;
    int nSamples_;
    int shadowValue_;
    float shadowThreshold_;
    
    bool initialized_;
    int frameCount_;
    int width_, height_, channels_;
    
    std::vector<float> samples_;
    std::vector<uint8_t> background_;
};

// Factory functions
inline BackgroundSubtractorMOG2 createBackgroundSubtractorMOG2(
    int history = 500, float varThreshold = 16.0f, bool detectShadows = true) {
    return BackgroundSubtractorMOG2(history, varThreshold, detectShadows);
}

inline BackgroundSubtractorKNN createBackgroundSubtractorKNN(
    int history = 500, float dist2Threshold = 400.0f, bool detectShadows = true) {
    return BackgroundSubtractorKNN(history, dist2Threshold, detectShadows);
}

} // namespace video
} // namespace neurova

#endif // NEUROVA_VIDEO_BACKGROUND_HPP
