// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file video/trackers.hpp
 * @brief Object tracking algorithms
 */

#ifndef NEUROVA_VIDEO_TRACKERS_HPP
#define NEUROVA_VIDEO_TRACKERS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <complex>
#include <memory>

namespace neurova {
namespace video {

// ============================================================================
// Bounding Box
// ============================================================================

struct Rect {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    
    Rect() = default;
    Rect(int x_, int y_, int w_, int h_) : x(x_), y(y_), width(w_), height(h_) {}
};

// ============================================================================
// Tracker Base Class
// ============================================================================

class Tracker {
public:
    virtual ~Tracker() = default;
    
    virtual bool init(const uint8_t* image, int w, int h, int channels, const Rect& bbox) = 0;
    virtual bool update(const uint8_t* image, int w, int h, int channels, Rect& bbox) = 0;
};

// ============================================================================
// Helper Functions
// ============================================================================

namespace detail {

inline std::vector<float> toGrayscale(const uint8_t* img, int w, int h, int channels) {
    std::vector<float> gray(w * h);
    if (channels == 1) {
        for (int i = 0; i < w * h; ++i) {
            gray[i] = static_cast<float>(img[i]);
        }
    } else {
        for (int i = 0; i < w * h; ++i) {
            float sum = 0;
            for (int c = 0; c < channels; ++c) {
                sum += img[i * channels + c];
            }
            gray[i] = sum / channels;
        }
    }
    return gray;
}

inline std::vector<float> extractPatch(const std::vector<float>& img, int imgW, int imgH,
                                       int x, int y, int patchW, int patchH) {
    std::vector<float> patch(patchW * patchH);
    for (int py = 0; py < patchH; ++py) {
        for (int px = 0; px < patchW; ++px) {
            int sx = std::clamp(x + px, 0, imgW - 1);
            int sy = std::clamp(y + py, 0, imgH - 1);
            patch[py * patchW + px] = img[sy * imgW + sx];
        }
    }
    return patch;
}

inline float normalizedCrossCorrelation(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0;
    
    float meanA = 0, meanB = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        meanA += a[i];
        meanB += b[i];
    }
    meanA /= a.size();
    meanB /= b.size();
    
    float num = 0, denA = 0, denB = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float da = a[i] - meanA;
        float db = b[i] - meanB;
        num += da * db;
        denA += da * da;
        denB += db * db;
    }
    
    if (denA < 1e-6f || denB < 1e-6f) return 0;
    return num / (std::sqrt(denA) * std::sqrt(denB));
}

// Simple 2D FFT (radix-2, in-place, for power-of-2 sizes)
inline void fft2D(std::vector<std::complex<float>>& data, int w, int h, bool inverse = false) {
    // Row FFT
    for (int y = 0; y < h; ++y) {
        // Simple DFT for non-power-of-2 sizes
        std::vector<std::complex<float>> row(w);
        for (int x = 0; x < w; ++x) row[x] = data[y * w + x];
        
        std::vector<std::complex<float>> out(w);
        float sign = inverse ? 1.0f : -1.0f;
        for (int k = 0; k < w; ++k) {
            std::complex<float> sum(0, 0);
            for (int n = 0; n < w; ++n) {
                float angle = sign * 2 * 3.14159265f * k * n / w;
                sum += row[n] * std::complex<float>(std::cos(angle), std::sin(angle));
            }
            out[k] = inverse ? sum / static_cast<float>(w) : sum;
        }
        for (int x = 0; x < w; ++x) data[y * w + x] = out[x];
    }
    
    // Column FFT
    for (int x = 0; x < w; ++x) {
        std::vector<std::complex<float>> col(h);
        for (int y = 0; y < h; ++y) col[y] = data[y * w + x];
        
        std::vector<std::complex<float>> out(h);
        float sign = inverse ? 1.0f : -1.0f;
        for (int k = 0; k < h; ++k) {
            std::complex<float> sum(0, 0);
            for (int n = 0; n < h; ++n) {
                float angle = sign * 2 * 3.14159265f * k * n / h;
                sum += col[n] * std::complex<float>(std::cos(angle), std::sin(angle));
            }
            out[k] = inverse ? sum / static_cast<float>(h) : sum;
        }
        for (int y = 0; y < h; ++y) data[y * w + x] = out[y];
    }
}

inline std::vector<float> createHanningWindow(int w, int h) {
    std::vector<float> window(w * h);
    for (int y = 0; y < h; ++y) {
        float wy = 0.5f * (1 - std::cos(2 * 3.14159265f * y / (h - 1)));
        for (int x = 0; x < w; ++x) {
            float wx = 0.5f * (1 - std::cos(2 * 3.14159265f * x / (w - 1)));
            window[y * w + x] = wx * wy;
        }
    }
    return window;
}

inline std::vector<float> createGaussianTarget(int w, int h, float sigma) {
    std::vector<float> target(w * h);
    float cx = w / 2.0f;
    float cy = h / 2.0f;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float dx = x - cx;
            float dy = y - cy;
            target[y * w + x] = std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
        }
    }
    return target;
}

} // namespace detail

// ============================================================================
// TrackerMIL - Multiple Instance Learning
// ============================================================================

class TrackerMIL : public Tracker {
public:
    struct Params {
        float samplerInitInRadius = 3.0f;
        int samplerInitMaxNegNum = 65;
        float samplerSearchWinSize = 25.0f;
        float samplerTrackInRadius = 4.0f;
        int samplerTrackMaxPosNum = 100000;
        int samplerTrackMaxNegNum = 65;
        int featureSetNumFeatures = 250;
    };
    
    TrackerMIL() : initialized_(false), searchWindow_(25) {}
    
    static std::unique_ptr<TrackerMIL> create() {
        return std::make_unique<TrackerMIL>();
    }
    
    bool init(const uint8_t* image, int w, int h, int channels, const Rect& bbox) override {
        auto gray = detail::toGrayscale(image, w, h, channels);
        
        int x = std::clamp(bbox.x, 0, w - 1);
        int y = std::clamp(bbox.y, 0, h - 1);
        int bw = std::min(bbox.width, w - x);
        int bh = std::min(bbox.height, h - y);
        
        if (bw <= 0 || bh <= 0) return false;
        
        template_ = detail::extractPatch(gray, w, h, x, y, bw, bh);
        templateW_ = bw;
        templateH_ = bh;
        bbox_ = Rect(x, y, bw, bh);
        imgW_ = w;
        imgH_ = h;
        initialized_ = true;
        
        return true;
    }
    
    bool update(const uint8_t* image, int w, int h, int channels, Rect& bbox) override {
        if (!initialized_) {
            bbox = bbox_;
            return false;
        }
        
        auto gray = detail::toGrayscale(image, w, h, channels);
        
        int searchX = std::max(0, bbox_.x - searchWindow_);
        int searchY = std::max(0, bbox_.y - searchWindow_);
        int searchX2 = std::min(w, bbox_.x + bbox_.width + searchWindow_);
        int searchY2 = std::min(h, bbox_.y + bbox_.height + searchWindow_);
        
        float bestScore = -1;
        int bestX = bbox_.x, bestY = bbox_.y;
        
        for (int sy = searchY; sy < searchY2 - templateH_; sy += 2) {
            for (int sx = searchX; sx < searchX2 - templateW_; sx += 2) {
                auto patch = detail::extractPatch(gray, w, h, sx, sy, templateW_, templateH_);
                float score = detail::normalizedCrossCorrelation(template_, patch);
                
                if (score > bestScore) {
                    bestScore = score;
                    bestX = sx;
                    bestY = sy;
                }
            }
        }
        
        if (bestScore > 0.5f) {
            bbox_ = Rect(bestX, bestY, templateW_, templateH_);
            
            // Update template with learning rate
            auto newTemplate = detail::extractPatch(gray, w, h, bestX, bestY, templateW_, templateH_);
            for (size_t i = 0; i < template_.size(); ++i) {
                template_[i] = 0.9f * template_[i] + 0.1f * newTemplate[i];
            }
            
            bbox = bbox_;
            return true;
        }
        
        bbox = bbox_;
        return false;
    }
    
    Params params;

private:
    bool initialized_;
    int searchWindow_;
    std::vector<float> template_;
    int templateW_, templateH_;
    int imgW_, imgH_;
    Rect bbox_;
};

// ============================================================================
// TrackerKCF - Kernelized Correlation Filters
// ============================================================================

class TrackerKCF : public Tracker {
public:
    struct Params {
        float detectThresh = 0.5f;
        float sigma = 0.2f;
        float lambda = 0.0001f;
        float interpFactor = 0.075f;
        float outputSigmaFactor = 0.1f;
    };
    
    TrackerKCF() : initialized_(false) {}
    
    static std::unique_ptr<TrackerKCF> create() {
        return std::make_unique<TrackerKCF>();
    }
    
    bool init(const uint8_t* image, int w, int h, int channels, const Rect& bbox) override {
        auto gray = detail::toGrayscale(image, w, h, channels);
        
        int x = std::clamp(bbox.x, 0, w - 1);
        int y = std::clamp(bbox.y, 0, h - 1);
        int bw = std::min(bbox.width, w - x);
        int bh = std::min(bbox.height, h - y);
        
        if (bw <= 0 || bh <= 0) return false;
        
        template_ = detail::extractPatch(gray, w, h, x, y, bw, bh);
        templateW_ = bw;
        templateH_ = bh;
        bbox_ = Rect(x, y, bw, bh);
        imgW_ = w;
        imgH_ = h;
        
        // Create target
        float sigma = params.outputSigmaFactor * std::min(bw, bh);
        target_ = detail::createGaussianTarget(bw, bh, sigma);
        
        // Create window
        window_ = detail::createHanningWindow(bw, bh);
        
        // Train initial filter
        train();
        initialized_ = true;
        
        return true;
    }
    
    bool update(const uint8_t* image, int w, int h, int channels, Rect& bbox) override {
        if (!initialized_) {
            bbox = bbox_;
            return false;
        }
        
        auto gray = detail::toGrayscale(image, w, h, channels);
        
        int cx = bbox_.x + bbox_.width / 2;
        int cy = bbox_.y + bbox_.height / 2;
        int searchX = std::max(0, cx - bbox_.width);
        int searchY = std::max(0, cy - bbox_.height);
        
        auto search = detail::extractPatch(gray, w, h, searchX, searchY, templateW_, templateH_);
        
        // Get features
        auto z = getFeatures(search);
        
        // Compute correlation
        auto response = correlate(z);
        
        // Find peak
        int peakX = 0, peakY = 0;
        float peakVal = response[0];
        for (int y = 0; y < templateH_; ++y) {
            for (int x = 0; x < templateW_; ++x) {
                if (response[y * templateW_ + x] > peakVal) {
                    peakVal = response[y * templateW_ + x];
                    peakX = x;
                    peakY = y;
                }
            }
        }
        
        if (peakVal > params.detectThresh) {
            int dx = peakX - templateW_ / 2;
            int dy = peakY - templateH_ / 2;
            
            int newX = std::clamp(bbox_.x + dx, 0, w - bbox_.width);
            int newY = std::clamp(bbox_.y + dy, 0, h - bbox_.height);
            
            bbox_ = Rect(newX, newY, bbox_.width, bbox_.height);
            
            // Update template
            template_ = detail::extractPatch(gray, w, h, newX, newY, templateW_, templateH_);
            train();
            
            bbox = bbox_;
            return true;
        }
        
        bbox = bbox_;
        return false;
    }
    
    Params params;

private:
    void train() {
        auto f = getFeatures(template_);
        
        // FFT of features and target
        std::vector<std::complex<float>> F(templateW_ * templateH_);
        std::vector<std::complex<float>> T(templateW_ * templateH_);
        
        for (int i = 0; i < templateW_ * templateH_; ++i) {
            F[i] = std::complex<float>(f[i], 0);
            T[i] = std::complex<float>(target_[i], 0);
        }
        
        detail::fft2D(F, templateW_, templateH_);
        detail::fft2D(T, templateW_, templateH_);
        
        // Compute alpha = T / (F*conj(F) + lambda)
        alpha_.resize(templateW_ * templateH_);
        tmpl_ = f;
        
        for (int i = 0; i < templateW_ * templateH_; ++i) {
            auto k = F[i] * std::conj(F[i]);
            alpha_[i] = T[i] / (k + std::complex<float>(params.lambda, 0));
        }
    }
    
    std::vector<float> getFeatures(const std::vector<float>& patch) {
        std::vector<float> features(patch.size());
        
        float mean = 0;
        for (float v : patch) mean += v;
        mean /= patch.size();
        
        float stddev = 0;
        for (float v : patch) stddev += (v - mean) * (v - mean);
        stddev = std::sqrt(stddev / patch.size()) + 1e-5f;
        
        for (size_t i = 0; i < patch.size(); ++i) {
            features[i] = (patch[i] - mean) / stddev * window_[i];
        }
        
        return features;
    }
    
    std::vector<float> correlate(const std::vector<float>& z) {
        std::vector<std::complex<float>> Z(templateW_ * templateH_);
        for (int i = 0; i < templateW_ * templateH_; ++i) {
            Z[i] = std::complex<float>(z[i], 0);
        }
        
        detail::fft2D(Z, templateW_, templateH_);
        
        std::vector<std::complex<float>> response(templateW_ * templateH_);
        for (int i = 0; i < templateW_ * templateH_; ++i) {
            response[i] = alpha_[i] * Z[i];
        }
        
        detail::fft2D(response, templateW_, templateH_, true);
        
        std::vector<float> result(templateW_ * templateH_);
        for (int i = 0; i < templateW_ * templateH_; ++i) {
            result[i] = response[i].real();
        }
        
        return result;
    }
    
    bool initialized_;
    std::vector<float> template_;
    std::vector<float> target_;
    std::vector<float> window_;
    std::vector<float> tmpl_;
    std::vector<std::complex<float>> alpha_;
    int templateW_, templateH_;
    int imgW_, imgH_;
    Rect bbox_;
};

// ============================================================================
// TrackerCSRT - Channel and Spatial Reliability Tracker
// ============================================================================

class TrackerCSRT : public Tracker {
public:
    struct Params {
        bool useHog = true;
        bool useColorNames = true;
        bool useGray = true;
        std::string windowFunction = "hann";
        int templateSize = 200;
        float padding = 3.0f;
        float filterLr = 0.02f;
        float psrThreshold = 0.035f;
    };
    
    TrackerCSRT() : initialized_(false) {}
    
    static std::unique_ptr<TrackerCSRT> create() {
        return std::make_unique<TrackerCSRT>();
    }
    
    bool init(const uint8_t* image, int w, int h, int channels, const Rect& bbox) override {
        auto gray = detail::toGrayscale(image, w, h, channels);
        
        int x = std::clamp(bbox.x, 0, w - 1);
        int y = std::clamp(bbox.y, 0, h - 1);
        int bw = std::min(bbox.width, w - x);
        int bh = std::min(bbox.height, h - y);
        
        if (bw <= 0 || bh <= 0) return false;
        
        template_ = detail::extractPatch(gray, w, h, x, y, bw, bh);
        templateW_ = bw;
        templateH_ = bh;
        bbox_ = Rect(x, y, bw, bh);
        imgW_ = w;
        imgH_ = h;
        
        // Initialize filter
        initFilter();
        initialized_ = true;
        
        return true;
    }
    
    bool update(const uint8_t* image, int w, int h, int channels, Rect& bbox) override {
        if (!initialized_) {
            bbox = bbox_;
            return false;
        }
        
        auto gray = detail::toGrayscale(image, w, h, channels);
        
        int pad = static_cast<int>(std::max(bbox_.width, bbox_.height) * params.padding);
        int searchX = std::max(0, bbox_.x - pad);
        int searchY = std::max(0, bbox_.y - pad);
        int searchW = std::min(w - searchX, bbox_.width + 2 * pad);
        int searchH = std::min(h - searchY, bbox_.height + 2 * pad);
        
        auto searchRegion = detail::extractPatch(gray, w, h, searchX, searchY, templateW_, templateH_);
        
        // Get features
        auto z = getFeatures(searchRegion);
        
        // Compute response
        auto response = correlate(z);
        
        // Find peak
        int peakX = 0, peakY = 0;
        float peakVal = response[0];
        for (int y = 0; y < templateH_; ++y) {
            for (int x = 0; x < templateW_; ++x) {
                if (response[y * templateW_ + x] > peakVal) {
                    peakVal = response[y * templateW_ + x];
                    peakX = x;
                    peakY = y;
                }
            }
        }
        
        // Compute PSR
        float sidelobeMean = 0, sidelobeStd = 0;
        int count = 0;
        for (int y = 0; y < templateH_; ++y) {
            for (int x = 0; x < templateW_; ++x) {
                if (std::abs(x - peakX) > 5 || std::abs(y - peakY) > 5) {
                    sidelobeMean += response[y * templateW_ + x];
                    count++;
                }
            }
        }
        sidelobeMean /= count;
        
        for (int y = 0; y < templateH_; ++y) {
            for (int x = 0; x < templateW_; ++x) {
                if (std::abs(x - peakX) > 5 || std::abs(y - peakY) > 5) {
                    float d = response[y * templateW_ + x] - sidelobeMean;
                    sidelobeStd += d * d;
                }
            }
        }
        sidelobeStd = std::sqrt(sidelobeStd / count);
        
        float psr = (peakVal - sidelobeMean) / (sidelobeStd + 1e-5f);
        
        if (psr > params.psrThreshold * 100) {
            float scaleX = static_cast<float>(searchW) / templateW_;
            float scaleY = static_cast<float>(searchH) / templateH_;
            
            int dx = static_cast<int>((peakX - templateW_ / 2) * scaleX);
            int dy = static_cast<int>((peakY - templateH_ / 2) * scaleY);
            
            int newX = std::clamp(bbox_.x + dx, 0, w - bbox_.width);
            int newY = std::clamp(bbox_.y + dy, 0, h - bbox_.height);
            
            bbox_ = Rect(newX, newY, bbox_.width, bbox_.height);
            
            // Update filter
            template_ = detail::extractPatch(gray, w, h, newX, newY, templateW_, templateH_);
            updateFilter();
            
            bbox = bbox_;
            return true;
        }
        
        bbox = bbox_;
        return false;
    }
    
    Params params;

private:
    void initFilter() {
        float sigma = 0.1f * std::min(templateW_, templateH_);
        target_ = detail::createGaussianTarget(templateW_, templateH_, sigma);
        window_ = detail::createHanningWindow(templateW_, templateH_);
        
        auto f = getFeatures(template_);
        
        std::vector<std::complex<float>> F(templateW_ * templateH_);
        std::vector<std::complex<float>> T(templateW_ * templateH_);
        
        for (int i = 0; i < templateW_ * templateH_; ++i) {
            F[i] = std::complex<float>(f[i], 0);
            T[i] = std::complex<float>(target_[i], 0);
        }
        
        detail::fft2D(F, templateW_, templateH_);
        detail::fft2D(T, templateW_, templateH_);
        
        filter_.resize(templateW_ * templateH_);
        model_ = F;
        
        for (int i = 0; i < templateW_ * templateH_; ++i) {
            filter_[i] = T[i] / (std::conj(F[i]) * F[i] + std::complex<float>(1e-4f, 0));
        }
    }
    
    void updateFilter() {
        auto f = getFeatures(template_);
        
        std::vector<std::complex<float>> F(templateW_ * templateH_);
        std::vector<std::complex<float>> T(templateW_ * templateH_);
        
        for (int i = 0; i < templateW_ * templateH_; ++i) {
            F[i] = std::complex<float>(f[i], 0);
            T[i] = std::complex<float>(target_[i], 0);
        }
        
        detail::fft2D(F, templateW_, templateH_);
        detail::fft2D(T, templateW_, templateH_);
        
        float lr = params.filterLr;
        for (int i = 0; i < templateW_ * templateH_; ++i) {
            auto newFilter = T[i] / (std::conj(F[i]) * F[i] + std::complex<float>(1e-4f, 0));
            filter_[i] = filter_[i] * (1.0f - lr) + newFilter * lr;
            model_[i] = model_[i] * (1.0f - lr) + F[i] * lr;
        }
    }
    
    std::vector<float> getFeatures(const std::vector<float>& patch) {
        std::vector<float> features(patch.size());
        
        float mean = 0;
        for (float v : patch) mean += v;
        mean /= patch.size();
        
        float stddev = 0;
        for (float v : patch) stddev += (v - mean) * (v - mean);
        stddev = std::sqrt(stddev / patch.size()) + 1e-5f;
        
        for (size_t i = 0; i < patch.size(); ++i) {
            features[i] = (patch[i] - mean) / stddev * window_[i];
        }
        
        return features;
    }
    
    std::vector<float> correlate(const std::vector<float>& z) {
        std::vector<std::complex<float>> Z(templateW_ * templateH_);
        for (int i = 0; i < templateW_ * templateH_; ++i) {
            Z[i] = std::complex<float>(z[i], 0);
        }
        
        detail::fft2D(Z, templateW_, templateH_);
        
        std::vector<std::complex<float>> response(templateW_ * templateH_);
        for (int i = 0; i < templateW_ * templateH_; ++i) {
            response[i] = filter_[i] * Z[i];
        }
        
        detail::fft2D(response, templateW_, templateH_, true);
        
        std::vector<float> result(templateW_ * templateH_);
        for (int i = 0; i < templateW_ * templateH_; ++i) {
            result[i] = response[i].real();
        }
        
        return result;
    }
    
    bool initialized_;
    std::vector<float> template_;
    std::vector<float> target_;
    std::vector<float> window_;
    std::vector<std::complex<float>> filter_;
    std::vector<std::complex<float>> model_;
    int templateW_, templateH_;
    int imgW_, imgH_;
    Rect bbox_;
};

// ============================================================================
// Factory Functions (Legacy API)
// ============================================================================

inline std::unique_ptr<TrackerMIL> TrackerMIL_create() {
    return TrackerMIL::create();
}

inline std::unique_ptr<TrackerKCF> TrackerKCF_create() {
    return TrackerKCF::create();
}

inline std::unique_ptr<TrackerCSRT> TrackerCSRT_create() {
    return TrackerCSRT::create();
}

} // namespace video
} // namespace neurova

#endif // NEUROVA_VIDEO_TRACKERS_HPP
