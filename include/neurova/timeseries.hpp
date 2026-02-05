// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file timeseries.hpp
 * @brief Time series analysis and forecasting
 */

#ifndef NEUROVA_TIMESERIES_HPP
#define NEUROVA_TIMESERIES_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace neurova {
namespace timeseries {

// ============================================================================
// Result Structures
// ============================================================================

/**
 * @brief Result from seasonal decomposition
 */
struct DecompositionResult {
    std::vector<float> observed;
    std::vector<float> trend;
    std::vector<float> seasonal;
    std::vector<float> residual;
};

/**
 * @brief Result from Augmented Dickey-Fuller test
 */
struct ADFResult {
    float statistic;
    float pvalue;
    int usedLag;
    int nobs;
    float criticalValue1;  // 1%
    float criticalValue5;  // 5%
    float criticalValue10; // 10%
};

// ============================================================================
// Statistical Functions
// ============================================================================

/**
 * @brief Compute autocorrelation function
 */
inline std::vector<float> acf(const float* x, int n, int nlags = 40) {
    nlags = std::min(nlags, n - 1);
    std::vector<float> result(nlags + 1);
    
    // Compute mean
    float mean = 0;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    
    // Compute variance
    float c0 = 0;
    for (int i = 0; i < n; ++i) {
        float diff = x[i] - mean;
        c0 += diff * diff;
    }
    c0 /= n;
    
    if (c0 < 1e-10f) {
        std::fill(result.begin(), result.end(), 0);
        result[0] = 1.0f;
        return result;
    }
    
    result[0] = 1.0f;
    for (int k = 1; k <= nlags; ++k) {
        float ck = 0;
        for (int i = 0; i < n - k; ++i) {
            ck += (x[i] - mean) * (x[i + k] - mean);
        }
        ck /= n;
        result[k] = ck / c0;
    }
    
    return result;
}

/**
 * @brief Compute partial autocorrelation function using Durbin-Levinson
 */
inline std::vector<float> pacf(const float* x, int n, int nlags = 40) {
    nlags = std::min(nlags, n - 1);
    
    auto acfVals = acf(x, n, nlags);
    
    std::vector<float> result(nlags + 1);
    result[0] = 1.0f;
    
    if (nlags < 1) return result;
    
    result[1] = acfVals[1];
    
    // Durbin-Levinson recursion
    std::vector<std::vector<float>> phi(nlags + 1, std::vector<float>(nlags + 1, 0));
    phi[1][1] = acfVals[1];
    
    for (int k = 2; k <= nlags; ++k) {
        float num = acfVals[k];
        for (int j = 1; j < k; ++j) {
            num -= phi[k-1][j] * acfVals[k - j];
        }
        
        float den = 1.0f;
        for (int j = 1; j < k; ++j) {
            den -= phi[k-1][j] * acfVals[j];
        }
        
        phi[k][k] = (std::abs(den) > 1e-10f) ? num / den : 0.0f;
        result[k] = phi[k][k];
        
        for (int j = 1; j < k; ++j) {
            phi[k][j] = phi[k-1][j] - phi[k][k] * phi[k-1][k - j];
        }
    }
    
    return result;
}

/**
 * @brief Augmented Dickey-Fuller test for unit root
 */
inline ADFResult adfuller(const float* x, int n, int maxlag = -1) {
    ADFResult result;
    
    if (maxlag < 0) {
        maxlag = static_cast<int>(std::ceil(12.0 * std::pow(n / 100.0, 0.25)));
    }
    
    // Compute first difference
    std::vector<float> dx(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        dx[i] = x[i + 1] - x[i];
    }
    
    // Simple regression: dx_t = alpha + gamma * x_{t-1} + epsilon
    int lag = std::min(maxlag, 1);
    int nReg = static_cast<int>(dx.size()) - lag;
    
    if (nReg < 3) {
        result.statistic = 0;
        result.pvalue = 1.0f;
        result.usedLag = lag;
        result.nobs = nReg;
        result.criticalValue1 = -3.51f;
        result.criticalValue5 = -2.89f;
        result.criticalValue10 = -2.58f;
        return result;
    }
    
    // Build y and X
    std::vector<float> y(nReg);
    std::vector<float> xLag(nReg);
    
    for (int i = 0; i < nReg; ++i) {
        y[i] = dx[i + lag];
        xLag[i] = x[i + lag];
    }
    
    // OLS: y = alpha + gamma * xLag
    float sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
    for (int i = 0; i < nReg; ++i) {
        sumX += xLag[i];
        sumY += y[i];
        sumXX += xLag[i] * xLag[i];
        sumXY += xLag[i] * y[i];
    }
    
    float meanX = sumX / nReg;
    float meanY = sumY / nReg;
    
    float sxx = sumXX - nReg * meanX * meanX;
    float sxy = sumXY - nReg * meanX * meanY;
    
    float gamma = (std::abs(sxx) > 1e-10f) ? sxy / sxx : 0.0f;
    float alpha = meanY - gamma * meanX;
    
    // Compute residuals and standard error
    float rss = 0;
    for (int i = 0; i < nReg; ++i) {
        float resid = y[i] - alpha - gamma * xLag[i];
        rss += resid * resid;
    }
    
    float se = std::sqrt(rss / (nReg - 2));
    float seGamma = (std::abs(sxx) > 1e-10f) ? se / std::sqrt(sxx) : 1.0f;
    
    result.statistic = (std::abs(seGamma) > 1e-10f) ? gamma / seGamma : 0.0f;
    result.usedLag = lag;
    result.nobs = nReg;
    
    // Critical values (approximate for constant regression)
    result.criticalValue1 = -3.51f;
    result.criticalValue5 = -2.89f;
    result.criticalValue10 = -2.58f;
    
    // Approximate p-value
    if (result.statistic < -3.5f) result.pvalue = 0.01f;
    else if (result.statistic < -2.9f) result.pvalue = 0.05f;
    else if (result.statistic < -2.6f) result.pvalue = 0.10f;
    else result.pvalue = 0.50f;
    
    return result;
}

// ============================================================================
// Decomposition
// ============================================================================

/**
 * @brief Seasonal decomposition using moving averages
 */
inline DecompositionResult seasonalDecompose(
    const float* y, int n,
    int period = 12,
    bool additive = true,
    bool extrapolateTrend = true) {
    
    DecompositionResult result;
    result.observed.assign(y, y + n);
    result.trend.resize(n, std::nanf(""));
    result.seasonal.resize(n);
    result.residual.resize(n);
    
    if (n < 2 * period) {
        throw std::invalid_argument("Time series too short for given period");
    }
    
    // Extract trend using centered moving average
    int halfPeriod = period / 2;
    
    if (period % 2 == 0) {
        // Double moving average for even period
        std::vector<float> ma1(n - period + 1);
        for (int i = 0; i < n - period + 1; ++i) {
            float sum = 0;
            for (int j = 0; j < period; ++j) {
                sum += y[i + j];
            }
            ma1[i] = sum / period;
        }
        
        // Second MA to center
        for (int i = 0; i < static_cast<int>(ma1.size()) - 1; ++i) {
            result.trend[halfPeriod + i] = (ma1[i] + ma1[i + 1]) / 2;
        }
    } else {
        // Simple centered MA for odd period
        for (int i = halfPeriod; i < n - halfPeriod; ++i) {
            float sum = 0;
            for (int j = -halfPeriod; j <= halfPeriod; ++j) {
                sum += y[i + j];
            }
            result.trend[i] = sum / period;
        }
    }
    
    // Extrapolate trend
    if (extrapolateTrend) {
        int firstValid = -1, lastValid = -1;
        for (int i = 0; i < n; ++i) {
            if (!std::isnan(result.trend[i])) {
                if (firstValid < 0) firstValid = i;
                lastValid = i;
            }
        }
        
        if (firstValid >= 0) {
            for (int i = 0; i < firstValid; ++i) {
                result.trend[i] = result.trend[firstValid];
            }
            for (int i = lastValid + 1; i < n; ++i) {
                result.trend[i] = result.trend[lastValid];
            }
        }
    }
    
    // Compute detrended series
    std::vector<float> detrended(n);
    for (int i = 0; i < n; ++i) {
        if (additive) {
            detrended[i] = y[i] - result.trend[i];
        } else {
            detrended[i] = (std::abs(result.trend[i]) > 1e-10f) ? y[i] / result.trend[i] : 1.0f;
        }
    }
    
    // Extract seasonal component by averaging
    std::vector<float> seasonalAvg(period, 0);
    std::vector<int> counts(period, 0);
    
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(detrended[i])) {
            seasonalAvg[i % period] += detrended[i];
            counts[i % period]++;
        }
    }
    
    for (int i = 0; i < period; ++i) {
        if (counts[i] > 0) {
            seasonalAvg[i] /= counts[i];
        }
    }
    
    // Center seasonal component
    if (additive) {
        float mean = 0;
        for (float v : seasonalAvg) mean += v;
        mean /= period;
        for (float& v : seasonalAvg) v -= mean;
    } else {
        float mean = 0;
        for (float v : seasonalAvg) mean += v;
        mean /= period;
        if (std::abs(mean) > 1e-10f) {
            for (float& v : seasonalAvg) v /= mean;
        }
    }
    
    // Repeat seasonal pattern
    for (int i = 0; i < n; ++i) {
        result.seasonal[i] = seasonalAvg[i % period];
    }
    
    // Compute residuals
    for (int i = 0; i < n; ++i) {
        if (additive) {
            result.residual[i] = y[i] - result.trend[i] - result.seasonal[i];
        } else {
            float denom = result.trend[i] * result.seasonal[i];
            result.residual[i] = (std::abs(denom) > 1e-10f) ? y[i] / denom : 1.0f;
        }
    }
    
    return result;
}

// ============================================================================
// Exponential Smoothing
// ============================================================================

/**
 * @brief Simple Exponential Smoothing
 */
class SimpleExponentialSmoothing {
public:
    SimpleExponentialSmoothing(float alpha = 0.5f) : alpha_(alpha), level_(0), fitted_(false) {
        if (alpha <= 0 || alpha >= 1) {
            throw std::invalid_argument("alpha must be between 0 and 1");
        }
    }
    
    void fit(const float* y, int n) {
        y_.assign(y, y + n);
        level_ = y[0];
        
        for (int t = 1; t < n; ++t) {
            level_ = alpha_ * y[t] + (1 - alpha_) * level_;
        }
        
        fitted_ = true;
    }
    
    std::vector<float> predict(int steps) const {
        if (!fitted_) {
            throw std::runtime_error("Model must be fitted before prediction");
        }
        return std::vector<float>(steps, level_);
    }
    
    float level() const { return level_; }
    
private:
    float alpha_;
    float level_;
    bool fitted_;
    std::vector<float> y_;
};

/**
 * @brief Holt-Winters Exponential Smoothing
 */
class ExponentialSmoothing {
public:
    enum class TrendType { NONE, ADDITIVE, MULTIPLICATIVE };
    enum class SeasonalType { NONE, ADDITIVE, MULTIPLICATIVE };
    
    ExponentialSmoothing(int seasonalPeriods = 12,
                        TrendType trend = TrendType::NONE,
                        SeasonalType seasonal = SeasonalType::NONE)
        : seasonalPeriods_(seasonalPeriods), trend_(trend), seasonal_(seasonal),
          level_(0), trendVal_(0), alpha_(0.3f), beta_(0.1f), gamma_(0.1f),
          fitted_(false) {}
    
    void fit(const float* y, int n) {
        if (seasonal_ != SeasonalType::NONE && n < 2 * seasonalPeriods_) {
            throw std::invalid_argument("Need more observations for seasonal model");
        }
        
        initializeComponents(y, n);
        
        for (int t = 0; t < n; ++t) {
            updateComponents(y[t], t);
        }
        
        fitted_ = true;
    }
    
    std::vector<float> predict(int steps) const {
        if (!fitted_) {
            throw std::runtime_error("Model must be fitted before prediction");
        }
        
        std::vector<float> forecasts(steps);
        
        for (int h = 0; h < steps; ++h) {
            float forecast = level_;
            
            if (trend_ == TrendType::ADDITIVE) {
                forecast += trendVal_ * (h + 1);
            } else if (trend_ == TrendType::MULTIPLICATIVE) {
                forecast *= std::pow(trendVal_, h + 1);
            }
            
            if (seasonal_ != SeasonalType::NONE && !seasonalVals_.empty()) {
                int sIdx = h % seasonalPeriods_;
                if (seasonal_ == SeasonalType::ADDITIVE) {
                    forecast += seasonalVals_[sIdx];
                } else {
                    forecast *= seasonalVals_[sIdx];
                }
            }
            
            forecasts[h] = forecast;
        }
        
        return forecasts;
    }
    
    void setAlpha(float alpha) { alpha_ = alpha; }
    void setBeta(float beta) { beta_ = beta; }
    void setGamma(float gamma) { gamma_ = gamma; }
    
private:
    void initializeComponents(const float* y, int n) {
        // Initialize level
        if (seasonal_ != SeasonalType::NONE) {
            float sum = 0;
            for (int i = 0; i < seasonalPeriods_; ++i) sum += y[i];
            level_ = sum / seasonalPeriods_;
        } else {
            level_ = y[0];
        }
        
        // Initialize trend
        if (trend_ != TrendType::NONE) {
            if (seasonal_ != SeasonalType::NONE && n >= 2 * seasonalPeriods_) {
                float sum1 = 0, sum2 = 0;
                for (int i = 0; i < seasonalPeriods_; ++i) {
                    sum1 += y[i];
                    sum2 += y[i + seasonalPeriods_];
                }
                trendVal_ = (sum2 - sum1) / (seasonalPeriods_ * seasonalPeriods_);
            } else {
                trendVal_ = (y[n-1] - y[0]) / (n - 1);
            }
        }
        
        // Initialize seasonal
        if (seasonal_ != SeasonalType::NONE) {
            seasonalVals_.resize(seasonalPeriods_);
            for (int i = 0; i < seasonalPeriods_; ++i) {
                std::vector<float> vals;
                for (int j = i; j < n; j += seasonalPeriods_) {
                    vals.push_back(y[j]);
                }
                float mean = 0;
                for (float v : vals) mean += v;
                mean /= vals.size();
                
                if (seasonal_ == SeasonalType::ADDITIVE) {
                    seasonalVals_[i] = mean - level_;
                } else {
                    seasonalVals_[i] = (std::abs(level_) > 1e-10f) ? mean / level_ : 1.0f;
                }
            }
        }
    }
    
    void updateComponents(float yt, int t) {
        float st = 0, levelNew, trendNew;
        
        if (seasonal_ != SeasonalType::NONE) {
            st = seasonalVals_[t % seasonalPeriods_];
        } else {
            st = (seasonal_ == SeasonalType::ADDITIVE) ? 0.0f : 1.0f;
        }
        
        // Update level
        if (seasonal_ == SeasonalType::ADDITIVE) {
            levelNew = alpha_ * (yt - st) + (1 - alpha_) * (level_ + trendVal_);
        } else if (seasonal_ == SeasonalType::MULTIPLICATIVE) {
            levelNew = alpha_ * (yt / st) + (1 - alpha_) * (level_ + trendVal_);
        } else {
            levelNew = alpha_ * yt + (1 - alpha_) * (level_ + trendVal_);
        }
        
        // Update trend
        if (trend_ != TrendType::NONE) {
            trendNew = beta_ * (levelNew - level_) + (1 - beta_) * trendVal_;
        } else {
            trendNew = 0;
        }
        
        // Update seasonal
        if (seasonal_ != SeasonalType::NONE) {
            int sIdx = t % seasonalPeriods_;
            if (seasonal_ == SeasonalType::ADDITIVE) {
                seasonalVals_[sIdx] = gamma_ * (yt - levelNew) + (1 - gamma_) * st;
            } else {
                seasonalVals_[sIdx] = gamma_ * (yt / levelNew) + (1 - gamma_) * st;
            }
        }
        
        level_ = levelNew;
        trendVal_ = trendNew;
    }
    
    int seasonalPeriods_;
    TrendType trend_;
    SeasonalType seasonal_;
    float level_;
    float trendVal_;
    std::vector<float> seasonalVals_;
    float alpha_, beta_, gamma_;
    bool fitted_;
};

// ============================================================================
// ARIMA
// ============================================================================

/**
 * @brief ARIMA(p, d, q) model
 */
class ARIMA {
public:
    ARIMA(int p = 1, int d = 0, int q = 0) : p_(p), d_(d), q_(q), const_(0), fitted_(false) {}
    
    void fit(const float* y, int n, int maxIter = 100) {
        yOriginal_.assign(y, y + n);
        
        // Apply differencing
        yDiff_ = difference(yOriginal_, d_);
        
        if (static_cast<int>(yDiff_.size()) < std::max(p_, q_) + 10) {
            throw std::invalid_argument("Not enough observations after differencing");
        }
        
        // Fit parameters
        if (p_ > 0 && q_ == 0) {
            fitAR();
        } else if (q_ > 0 && p_ == 0) {
            fitMA();
        } else {
            fitARMA(maxIter);
        }
        
        // Calculate residuals
        residuals_ = calculateResiduals();
        fitted_ = true;
    }
    
    std::vector<float> predict(int steps) const {
        if (!fitted_) {
            throw std::runtime_error("Model must be fitted before prediction");
        }
        
        std::vector<float> forecasts(steps);
        std::vector<float> history = yDiff_;
        std::vector<float> errors = residuals_;
        
        for (int h = 0; h < steps; ++h) {
            float forecast = const_;
            
            // AR part
            for (int i = 0; i < p_ && i < static_cast<int>(history.size()); ++i) {
                forecast += arParams_[i] * history[history.size() - 1 - i];
            }
            
            // MA part
            for (int i = 0; i < q_ && i < static_cast<int>(errors.size()); ++i) {
                forecast += maParams_[i] * errors[errors.size() - 1 - i];
            }
            
            forecasts[h] = forecast;
            history.push_back(forecast);
            errors.push_back(0);  // Assume future errors are 0
        }
        
        // Undo differencing
        return undifference(forecasts, yOriginal_, d_);
    }
    
    const std::vector<float>& residuals() const { return residuals_; }
    const std::vector<float>& arParams() const { return arParams_; }
    const std::vector<float>& maParams() const { return maParams_; }
    
private:
    std::vector<float> difference(const std::vector<float>& y, int d) const {
        std::vector<float> result = y;
        for (int i = 0; i < d; ++i) {
            std::vector<float> diff(result.size() - 1);
            for (size_t j = 0; j < diff.size(); ++j) {
                diff[j] = result[j + 1] - result[j];
            }
            result = diff;
        }
        return result;
    }
    
    std::vector<float> undifference(const std::vector<float>& yDiff,
                                   const std::vector<float>& yOrig, int d) const {
        if (d == 0) return yDiff;
        
        std::vector<float> result = yDiff;
        for (int i = 0; i < d; ++i) {
            std::vector<float> cumsum(result.size() + 1);
            cumsum[0] = yOrig.back();
            for (size_t j = 0; j < result.size(); ++j) {
                cumsum[j + 1] = cumsum[j] + result[j];
            }
            result.assign(cumsum.begin() + 1, cumsum.end());
        }
        return result;
    }
    
    void fitAR() {
        int n = static_cast<int>(yDiff_.size());
        
        // Compute mean
        float mean = 0;
        for (float v : yDiff_) mean += v;
        mean /= n;
        const_ = mean;
        
        // Center data
        std::vector<float> centered(n);
        for (int i = 0; i < n; ++i) {
            centered[i] = yDiff_[i] - mean;
        }
        
        // Compute ACF
        auto acfVals = acf(centered.data(), n, p_);
        
        // Yule-Walker equations
        std::vector<float> R(p_ * p_, 0);
        for (int i = 0; i < p_; ++i) {
            for (int j = 0; j < p_; ++j) {
                R[i * p_ + j] = acfVals[std::abs(i - j)];
            }
        }
        
        std::vector<float> r(p_);
        for (int i = 0; i < p_; ++i) {
            r[i] = acfVals[i + 1];
        }
        
        // Solve using simple Gaussian elimination
        arParams_.resize(p_);
        solveLinearSystem(R.data(), r.data(), arParams_.data(), p_);
        
        maParams_.clear();
    }
    
    void fitMA() {
        int n = static_cast<int>(yDiff_.size());
        
        float mean = 0;
        for (float v : yDiff_) mean += v;
        mean /= n;
        const_ = mean;
        
        std::vector<float> centered(n);
        for (int i = 0; i < n; ++i) {
            centered[i] = yDiff_[i] - mean;
        }
        
        // Approximate MA from ACF
        auto acfVals = acf(centered.data(), n, q_);
        
        maParams_.resize(q_);
        for (int i = 0; i < q_; ++i) {
            maParams_[i] = -acfVals[i + 1] * 0.5f;
        }
        
        arParams_.clear();
    }
    
    void fitARMA(int maxIter) {
        int n = static_cast<int>(yDiff_.size());
        
        float mean = 0;
        for (float v : yDiff_) mean += v;
        mean /= n;
        const_ = mean;
        
        // Initialize parameters
        arParams_.resize(p_, 0.1f);
        maParams_.resize(q_, 0.1f);
        
        // Simple gradient descent
        float lr = 0.01f;
        
        for (int iter = 0; iter < maxIter; ++iter) {
            auto errors = calculateResidualsARMA(arParams_, maParams_);
            
            // Update AR parameters
            for (int i = 0; i < p_; ++i) {
                float grad = 0;
                for (int t = std::max(p_, q_); t < static_cast<int>(yDiff_.size()); ++t) {
                    if (t - i - 1 >= 0) {
                        grad -= 2 * errors[t] * (yDiff_[t - i - 1] - const_);
                    }
                }
                grad /= (yDiff_.size() - std::max(p_, q_));
                arParams_[i] -= lr * std::clamp(grad, -1.0f, 1.0f);
            }
            
            // Update MA parameters
            for (int i = 0; i < q_; ++i) {
                float grad = 0;
                for (int t = std::max(p_, q_); t < static_cast<int>(errors.size()); ++t) {
                    if (t - i - 1 >= 0) {
                        grad -= 2 * errors[t] * errors[t - i - 1];
                    }
                }
                grad /= (errors.size() - std::max(p_, q_));
                maParams_[i] -= lr * std::clamp(grad, -1.0f, 1.0f);
            }
        }
    }
    
    std::vector<float> calculateResiduals() const {
        return calculateResidualsARMA(arParams_, maParams_);
    }
    
    std::vector<float> calculateResidualsARMA(
        const std::vector<float>& ar,
        const std::vector<float>& ma) const {
        
        int n = static_cast<int>(yDiff_.size());
        std::vector<float> errors(n, 0);
        
        for (int t = 0; t < n; ++t) {
            float pred = const_;
            
            for (int i = 0; i < p_ && t - i - 1 >= 0; ++i) {
                pred += ar[i] * (yDiff_[t - i - 1] - const_);
            }
            
            for (int i = 0; i < q_ && t - i - 1 >= 0; ++i) {
                pred += ma[i] * errors[t - i - 1];
            }
            
            errors[t] = (yDiff_[t] - const_) - (pred - const_);
        }
        
        return errors;
    }
    
    void solveLinearSystem(const float* A, const float* b, float* x, int n) const {
        // Simple Gaussian elimination
        std::vector<float> augmented(n * (n + 1));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                augmented[i * (n + 1) + j] = A[i * n + j];
            }
            augmented[i * (n + 1) + n] = b[i];
        }
        
        // Forward elimination
        for (int k = 0; k < n; ++k) {
            // Find pivot
            int maxRow = k;
            for (int i = k + 1; i < n; ++i) {
                if (std::abs(augmented[i * (n + 1) + k]) > 
                    std::abs(augmented[maxRow * (n + 1) + k])) {
                    maxRow = i;
                }
            }
            
            // Swap rows
            for (int j = k; j <= n; ++j) {
                std::swap(augmented[k * (n + 1) + j], augmented[maxRow * (n + 1) + j]);
            }
            
            // Eliminate column
            for (int i = k + 1; i < n; ++i) {
                float factor = augmented[i * (n + 1) + k] / 
                              (augmented[k * (n + 1) + k] + 1e-10f);
                for (int j = k; j <= n; ++j) {
                    augmented[i * (n + 1) + j] -= factor * augmented[k * (n + 1) + j];
                }
            }
        }
        
        // Back substitution
        for (int i = n - 1; i >= 0; --i) {
            x[i] = augmented[i * (n + 1) + n];
            for (int j = i + 1; j < n; ++j) {
                x[i] -= augmented[i * (n + 1) + j] * x[j];
            }
            x[i] /= (augmented[i * (n + 1) + i] + 1e-10f);
        }
    }
    
    int p_, d_, q_;
    float const_;
    std::vector<float> arParams_;
    std::vector<float> maParams_;
    std::vector<float> yOriginal_;
    std::vector<float> yDiff_;
    std::vector<float> residuals_;
    bool fitted_;
};

} // namespace timeseries
} // namespace neurova

#endif // NEUROVA_TIMESERIES_HPP
