// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file video/tracking.hpp
 * @brief Tracking algorithms: MeanShift, CamShift, KalmanFilter
 */

#ifndef NEUROVA_VIDEO_TRACKING_HPP
#define NEUROVA_VIDEO_TRACKING_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <array>

namespace neurova {
namespace video {

// ============================================================================
// Termination Criteria
// ============================================================================

enum TermCriteria {
    TERM_CRITERIA_EPS = 1,
    TERM_CRITERIA_MAX_ITER = 2,
    TERM_CRITERIA_COUNT = 2
};

struct Criteria {
    int type = TERM_CRITERIA_MAX_ITER | TERM_CRITERIA_EPS;
    int maxCount = 10;
    float epsilon = 1.0f;
    
    Criteria() = default;
    Criteria(int type_, int maxCount_, float epsilon_)
        : type(type_), maxCount(maxCount_), epsilon(epsilon_) {}
};

// ============================================================================
// Rotated Rect (for CamShift result)
// ============================================================================

struct RotatedRect {
    float centerX = 0;
    float centerY = 0;
    float width = 0;
    float height = 0;
    float angle = 0;
    
    RotatedRect() = default;
    RotatedRect(float cx, float cy, float w, float h, float a)
        : centerX(cx), centerY(cy), width(w), height(h), angle(a) {}
};

// ============================================================================
// Image Moments
// ============================================================================

struct Moments {
    float m00 = 0, m10 = 0, m01 = 0;
    float m20 = 0, m02 = 0, m11 = 0;
    float m30 = 0, m03 = 0, m12 = 0, m21 = 0;
};

inline Moments computeMoments(const float* roi, int w, int h) {
    Moments M;
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = roi[y * w + x];
            M.m00 += v;
            M.m10 += x * v;
            M.m01 += y * v;
            M.m20 += x * x * v;
            M.m02 += y * y * v;
            M.m11 += x * y * v;
        }
    }
    
    return M;
}

// ============================================================================
// MeanShift
// ============================================================================

struct MeanShiftResult {
    int iterations = 0;
    int x = 0, y = 0, width = 0, height = 0;
};

/**
 * @brief Find object on a back projection image using meanshift
 * 
 * @param probImage Back projection probability image
 * @param w Image width
 * @param h Image height
 * @param windowX Initial window x
 * @param windowY Initial window y
 * @param windowW Window width
 * @param windowH Window height
 * @param criteria Termination criteria
 * @return MeanShiftResult with iterations and updated window
 */
inline MeanShiftResult meanShift(
    const uint8_t* probImage, int imgW, int imgH,
    int windowX, int windowY, int windowW, int windowH,
    const Criteria& criteria = Criteria())
{
    MeanShiftResult result;
    result.x = windowX;
    result.y = windowY;
    result.width = windowW;
    result.height = windowH;
    
    int x = windowX;
    int y = windowY;
    int w = windowW;
    int h = windowH;
    
    for (int iter = 0; iter < criteria.maxCount; ++iter) {
        // Clamp window to image bounds
        x = std::max(0, std::min(x, imgW - w));
        y = std::max(0, std::min(y, imgH - h));
        
        // Compute center of mass
        float total = 0;
        float sumX = 0, sumY = 0;
        
        for (int wy = 0; wy < h; ++wy) {
            for (int wx = 0; wx < w; ++wx) {
                int px = x + wx;
                int py = y + wy;
                if (px >= 0 && px < imgW && py >= 0 && py < imgH) {
                    float val = static_cast<float>(probImage[py * imgW + px]);
                    total += val;
                    sumX += wx * val;
                    sumY += wy * val;
                }
            }
        }
        
        if (total < 1e-10f) break;
        
        float cx = sumX / total;
        float cy = sumY / total;
        
        // New center in image coordinates
        int newX = static_cast<int>(x + cx - w / 2.0f);
        int newY = static_cast<int>(y + cy - h / 2.0f);
        
        // Check convergence
        int dx = newX - x;
        int dy = newY - y;
        
        result.iterations = iter + 1;
        
        if (dx * dx + dy * dy < criteria.epsilon * criteria.epsilon) {
            result.x = newX;
            result.y = newY;
            result.width = w;
            result.height = h;
            return result;
        }
        
        x = newX;
        y = newY;
    }
    
    result.x = x;
    result.y = y;
    result.width = w;
    result.height = h;
    return result;
}

// ============================================================================
// CamShift
// ============================================================================

struct CamShiftResult {
    RotatedRect rotatedRect;
    int windowX = 0, windowY = 0, windowW = 0, windowH = 0;
};

/**
 * @brief Find object center, size, and orientation using CAMshift algorithm
 * 
 * @param probImage Back projection probability image
 * @param imgW Image width
 * @param imgH Image height
 * @param windowX Initial window x
 * @param windowY Initial window y
 * @param windowW Window width
 * @param windowH Window height
 * @param criteria Termination criteria
 * @return CamShiftResult with rotated rect and updated window
 */
inline CamShiftResult CamShift(
    const uint8_t* probImage, int imgW, int imgH,
    int windowX, int windowY, int windowW, int windowH,
    const Criteria& criteria = Criteria())
{
    CamShiftResult result;
    
    // Run meanShift first
    auto msResult = meanShift(probImage, imgW, imgH, windowX, windowY, windowW, windowH, criteria);
    
    int x = msResult.x;
    int y = msResult.y;
    int w = msResult.width;
    int h = msResult.height;
    
    // Clamp to image bounds
    x = std::max(0, std::min(x, imgW - 1));
    y = std::max(0, std::min(y, imgH - 1));
    w = std::max(1, std::min(w, imgW - x));
    h = std::max(1, std::min(h, imgH - y));
    
    // Extract ROI and compute moments
    std::vector<float> roi(w * h);
    for (int wy = 0; wy < h; ++wy) {
        for (int wx = 0; wx < w; ++wx) {
            roi[wy * w + wx] = static_cast<float>(probImage[(y + wy) * imgW + (x + wx)]);
        }
    }
    
    auto M = computeMoments(roi.data(), w, h);
    
    if (M.m00 > 0) {
        // Center of mass within ROI
        float cx = M.m10 / M.m00;
        float cy = M.m01 / M.m00;
        
        // Compute orientation from second moments
        float mu20 = M.m20 / M.m00 - cx * cx;
        float mu02 = M.m02 / M.m00 - cy * cy;
        float mu11 = M.m11 / M.m00 - cx * cy;
        
        // Orientation angle
        float angle = 0;
        if (std::abs(mu20 - mu02) > 1e-10f) {
            angle = 0.5f * std::atan2(2 * mu11, mu20 - mu02) * 180.0f / 3.14159265f;
        }
        
        // Adjust window size based on moments
        float s = 2.0f * std::sqrt(M.m00);
        int newW = std::max(1, static_cast<int>(s * 1.2f));
        int newH = std::max(1, static_cast<int>(s * 1.2f));
        
        // Center in image coordinates
        float centerX = x + cx;
        float centerY = y + cy;
        
        // Update window
        int newX = std::max(0, static_cast<int>(centerX - newW / 2.0f));
        int newY = std::max(0, static_cast<int>(centerY - newH / 2.0f));
        newW = std::min(newW, imgW - newX);
        newH = std::min(newH, imgH - newY);
        
        result.rotatedRect = RotatedRect(centerX, centerY, 
                                         static_cast<float>(newW), 
                                         static_cast<float>(newH), angle);
        result.windowX = newX;
        result.windowY = newY;
        result.windowW = newW;
        result.windowH = newH;
    } else {
        // Fallback
        result.rotatedRect = RotatedRect(x + w / 2.0f, y + h / 2.0f, 
                                         static_cast<float>(w), 
                                         static_cast<float>(h), 0);
        result.windowX = x;
        result.windowY = y;
        result.windowW = w;
        result.windowH = h;
    }
    
    return result;
}

// ============================================================================
// Kalman Filter
// ============================================================================

class KalmanFilter {
public:
    /**
     * @brief Construct Kalman filter
     * 
     * @param dynamParams Dimensionality of the state
     * @param measureParams Dimensionality of the measurement
     * @param controlParams Dimensionality of the control vector (default 0)
     */
    KalmanFilter(int dynamParams, int measureParams, int controlParams = 0)
        : dynamParams_(dynamParams), measureParams_(measureParams), controlParams_(controlParams)
    {
        // State vectors
        statePre_.resize(dynamParams, 0.0f);
        statePost_.resize(dynamParams, 0.0f);
        
        // Transition matrix (A) - identity by default
        transitionMatrix_.resize(dynamParams * dynamParams, 0.0f);
        for (int i = 0; i < dynamParams; ++i) {
            transitionMatrix_[i * dynamParams + i] = 1.0f;
        }
        
        // Control matrix (B)
        int ctrlDim = std::max(controlParams, 1);
        controlMatrix_.resize(dynamParams * ctrlDim, 0.0f);
        
        // Measurement matrix (H)
        measurementMatrix_.resize(measureParams * dynamParams, 0.0f);
        
        // Process noise covariance (Q) - identity
        processNoiseCov_.resize(dynamParams * dynamParams, 0.0f);
        for (int i = 0; i < dynamParams; ++i) {
            processNoiseCov_[i * dynamParams + i] = 1.0f;
        }
        
        // Measurement noise covariance (R) - identity
        measurementNoiseCov_.resize(measureParams * measureParams, 0.0f);
        for (int i = 0; i < measureParams; ++i) {
            measurementNoiseCov_[i * measureParams + i] = 1.0f;
        }
        
        // Error covariance matrices
        errorCovPre_.resize(dynamParams * dynamParams, 0.0f);
        errorCovPost_.resize(dynamParams * dynamParams, 0.0f);
        
        // Kalman gain
        gain_.resize(dynamParams * measureParams, 0.0f);
    }
    
    /**
     * @brief Compute predicted state
     * 
     * @param control Optional control vector
     * @return Predicted state vector
     */
    std::vector<float> predict(const std::vector<float>& control = {}) {
        // statePre = A * statePost
        matMul(transitionMatrix_.data(), statePost_.data(), statePre_.data(),
               dynamParams_, dynamParams_, 1);
        
        // Add control input
        if (!control.empty() && controlParams_ > 0) {
            std::vector<float> Bu(dynamParams_, 0.0f);
            matMul(controlMatrix_.data(), control.data(), Bu.data(),
                   dynamParams_, controlParams_, 1);
            for (int i = 0; i < dynamParams_; ++i) {
                statePre_[i] += Bu[i];
            }
        }
        
        // errorCovPre = A * errorCovPost * A' + Q
        std::vector<float> temp(dynamParams_ * dynamParams_);
        matMul(transitionMatrix_.data(), errorCovPost_.data(), temp.data(),
               dynamParams_, dynamParams_, dynamParams_);
        
        std::vector<float> At(dynamParams_ * dynamParams_);
        transpose(transitionMatrix_.data(), At.data(), dynamParams_, dynamParams_);
        
        matMul(temp.data(), At.data(), errorCovPre_.data(),
               dynamParams_, dynamParams_, dynamParams_);
        
        for (int i = 0; i < dynamParams_ * dynamParams_; ++i) {
            errorCovPre_[i] += processNoiseCov_[i];
        }
        
        statePost_ = statePre_;
        return statePre_;
    }
    
    /**
     * @brief Update predicted state from measurement
     * 
     * @param measurement Measured state vector
     * @return Corrected state vector
     */
    std::vector<float> correct(const std::vector<float>& measurement) {
        // temp2 = H * errorCovPre
        std::vector<float> temp2(measureParams_ * dynamParams_);
        matMul(measurementMatrix_.data(), errorCovPre_.data(), temp2.data(),
               measureParams_, dynamParams_, dynamParams_);
        
        // temp3 = H * errorCovPre * H' + R
        std::vector<float> Ht(dynamParams_ * measureParams_);
        transpose(measurementMatrix_.data(), Ht.data(), measureParams_, dynamParams_);
        
        std::vector<float> temp3(measureParams_ * measureParams_);
        matMul(temp2.data(), Ht.data(), temp3.data(),
               measureParams_, dynamParams_, measureParams_);
        
        for (int i = 0; i < measureParams_ * measureParams_; ++i) {
            temp3[i] += measurementNoiseCov_[i];
        }
        
        // Kalman gain: K = errorCovPre * H' * inv(temp3)
        std::vector<float> temp3Inv(measureParams_ * measureParams_);
        if (!invertMatrix(temp3.data(), temp3Inv.data(), measureParams_)) {
            // Use pseudo-inverse if singular
            temp3Inv = temp3;  // fallback
        }
        
        std::vector<float> PHt(dynamParams_ * measureParams_);
        matMul(errorCovPre_.data(), Ht.data(), PHt.data(),
               dynamParams_, dynamParams_, measureParams_);
        
        matMul(PHt.data(), temp3Inv.data(), gain_.data(),
               dynamParams_, measureParams_, measureParams_);
        
        // statePost = statePre + K * (measurement - H * statePre)
        std::vector<float> Hx(measureParams_);
        matMul(measurementMatrix_.data(), statePre_.data(), Hx.data(),
               measureParams_, dynamParams_, 1);
        
        std::vector<float> innovation(measureParams_);
        for (int i = 0; i < measureParams_; ++i) {
            innovation[i] = measurement[i] - Hx[i];
        }
        
        std::vector<float> KInno(dynamParams_);
        matMul(gain_.data(), innovation.data(), KInno.data(),
               dynamParams_, measureParams_, 1);
        
        for (int i = 0; i < dynamParams_; ++i) {
            statePost_[i] = statePre_[i] + KInno[i];
        }
        
        // errorCovPost = (I - K * H) * errorCovPre
        std::vector<float> KH(dynamParams_ * dynamParams_);
        matMul(gain_.data(), measurementMatrix_.data(), KH.data(),
               dynamParams_, measureParams_, dynamParams_);
        
        std::vector<float> I_KH(dynamParams_ * dynamParams_, 0.0f);
        for (int i = 0; i < dynamParams_; ++i) {
            I_KH[i * dynamParams_ + i] = 1.0f;
        }
        for (int i = 0; i < dynamParams_ * dynamParams_; ++i) {
            I_KH[i] -= KH[i];
        }
        
        matMul(I_KH.data(), errorCovPre_.data(), errorCovPost_.data(),
               dynamParams_, dynamParams_, dynamParams_);
        
        return statePost_;
    }
    
    /**
     * @brief Re-initialize the filter
     */
    void init(const std::vector<float>& statePre, const std::vector<float>& errorCovPost) {
        statePre_ = statePre;
        statePost_ = statePre;
        errorCovPost_ = errorCovPost;
        errorCovPre_ = errorCovPost;
    }
    
    // Public access to matrices for setup
    std::vector<float>& transitionMatrix() { return transitionMatrix_; }
    std::vector<float>& controlMatrix() { return controlMatrix_; }
    std::vector<float>& measurementMatrix() { return measurementMatrix_; }
    std::vector<float>& processNoiseCov() { return processNoiseCov_; }
    std::vector<float>& measurementNoiseCov() { return measurementNoiseCov_; }
    std::vector<float>& errorCovPost() { return errorCovPost_; }
    std::vector<float>& statePost() { return statePost_; }
    std::vector<float>& statePre() { return statePre_; }
    const std::vector<float>& gain() const { return gain_; }

private:
    // Matrix multiply: C = A * B (A: m x k, B: k x n, C: m x n)
    void matMul(const float* A, const float* B, float* C, int m, int k, int n) {
        std::fill(C, C + m * n, 0.0f);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int l = 0; l < k; ++l) {
                    C[i * n + j] += A[i * k + l] * B[l * n + j];
                }
            }
        }
    }
    
    // Transpose: B = A'
    void transpose(const float* A, float* B, int m, int n) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                B[j * m + i] = A[i * n + j];
            }
        }
    }
    
    // Simple matrix inversion (Gauss-Jordan for small matrices)
    bool invertMatrix(const float* A, float* Ainv, int n) {
        std::vector<float> aug(n * 2 * n);
        
        // Create augmented matrix [A | I]
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                aug[i * 2 * n + j] = A[i * n + j];
                aug[i * 2 * n + n + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        
        // Gaussian elimination
        for (int i = 0; i < n; ++i) {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < n; ++k) {
                if (std::abs(aug[k * 2 * n + i]) > std::abs(aug[maxRow * 2 * n + i])) {
                    maxRow = k;
                }
            }
            
            // Swap rows
            for (int k = 0; k < 2 * n; ++k) {
                std::swap(aug[i * 2 * n + k], aug[maxRow * 2 * n + k]);
            }
            
            float pivot = aug[i * 2 * n + i];
            if (std::abs(pivot) < 1e-10f) return false;
            
            // Scale row
            for (int k = 0; k < 2 * n; ++k) {
                aug[i * 2 * n + k] /= pivot;
            }
            
            // Eliminate column
            for (int k = 0; k < n; ++k) {
                if (k != i) {
                    float factor = aug[k * 2 * n + i];
                    for (int j = 0; j < 2 * n; ++j) {
                        aug[k * 2 * n + j] -= factor * aug[i * 2 * n + j];
                    }
                }
            }
        }
        
        // Extract inverse
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Ainv[i * n + j] = aug[i * 2 * n + n + j];
            }
        }
        
        return true;
    }
    
    int dynamParams_;
    int measureParams_;
    int controlParams_;
    
    std::vector<float> statePre_;
    std::vector<float> statePost_;
    std::vector<float> transitionMatrix_;
    std::vector<float> controlMatrix_;
    std::vector<float> measurementMatrix_;
    std::vector<float> processNoiseCov_;
    std::vector<float> measurementNoiseCov_;
    std::vector<float> errorCovPre_;
    std::vector<float> errorCovPost_;
    std::vector<float> gain_;
};

} // namespace video
} // namespace neurova

#endif // NEUROVA_VIDEO_TRACKING_HPP
