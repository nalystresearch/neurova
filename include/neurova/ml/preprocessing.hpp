// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file preprocessing.hpp
 * @brief Data preprocessing for machine learning
 */

#ifndef NEUROVA_ML_PREPROCESSING_HPP
#define NEUROVA_ML_PREPROCESSING_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <string>
#include <limits>

namespace neurova {
namespace ml {

/**
 * @brief Standard scaler - standardize features by removing mean and scaling to unit variance
 */
class StandardScaler {
public:
    StandardScaler() : fitted_(false) {}
    
    /**
     * @brief Fit the scaler on training data
     * @param data Input data [n_samples x n_features]
     * @param nSamples Number of samples
     * @param nFeatures Number of features
     */
    void fit(const float* data, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        mean_.resize(nFeatures, 0.0f);
        std_.resize(nFeatures, 0.0f);
        
        // Compute mean
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nFeatures; ++j) {
                mean_[j] += data[i * nFeatures + j];
            }
        }
        for (int j = 0; j < nFeatures; ++j) {
            mean_[j] /= nSamples;
        }
        
        // Compute standard deviation
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nFeatures; ++j) {
                float diff = data[i * nFeatures + j] - mean_[j];
                std_[j] += diff * diff;
            }
        }
        for (int j = 0; j < nFeatures; ++j) {
            std_[j] = std::sqrt(std_[j] / nSamples);
            if (std_[j] < 1e-10f) std_[j] = 1.0f;  // Avoid division by zero
        }
        
        fitted_ = true;
    }
    
    /**
     * @brief Transform data using fitted parameters
     */
    void transform(const float* input, float* output, int nSamples) const {
        if (!fitted_) return;
        
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nFeatures_; ++j) {
                output[i * nFeatures_ + j] = (input[i * nFeatures_ + j] - mean_[j]) / std_[j];
            }
        }
    }
    
    /**
     * @brief Fit and transform in one step
     */
    void fitTransform(const float* data, float* output, int nSamples, int nFeatures) {
        fit(data, nSamples, nFeatures);
        transform(data, output, nSamples);
    }
    
    /**
     * @brief Inverse transform
     */
    void inverseTransform(const float* input, float* output, int nSamples) const {
        if (!fitted_) return;
        
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nFeatures_; ++j) {
                output[i * nFeatures_ + j] = input[i * nFeatures_ + j] * std_[j] + mean_[j];
            }
        }
    }
    
    const std::vector<float>& mean() const { return mean_; }
    const std::vector<float>& std() const { return std_; }
    
private:
    std::vector<float> mean_;
    std::vector<float> std_;
    int nFeatures_ = 0;
    bool fitted_;
};

/**
 * @brief MinMax scaler - scale features to a given range [min, max]
 */
class MinMaxScaler {
public:
    MinMaxScaler(float featureMin = 0.0f, float featureMax = 1.0f)
        : featureMin_(featureMin), featureMax_(featureMax), fitted_(false) {}
    
    void fit(const float* data, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        dataMin_.resize(nFeatures, std::numeric_limits<float>::max());
        dataMax_.resize(nFeatures, std::numeric_limits<float>::lowest());
        
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nFeatures; ++j) {
                float val = data[i * nFeatures + j];
                dataMin_[j] = std::min(dataMin_[j], val);
                dataMax_[j] = std::max(dataMax_[j], val);
            }
        }
        
        fitted_ = true;
    }
    
    void transform(const float* input, float* output, int nSamples) const {
        if (!fitted_) return;
        
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nFeatures_; ++j) {
                float range = dataMax_[j] - dataMin_[j];
                if (range < 1e-10f) range = 1.0f;
                float scaled = (input[i * nFeatures_ + j] - dataMin_[j]) / range;
                output[i * nFeatures_ + j] = scaled * (featureMax_ - featureMin_) + featureMin_;
            }
        }
    }
    
    void fitTransform(const float* data, float* output, int nSamples, int nFeatures) {
        fit(data, nSamples, nFeatures);
        transform(data, output, nSamples);
    }
    
    void inverseTransform(const float* input, float* output, int nSamples) const {
        if (!fitted_) return;
        
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nFeatures_; ++j) {
                float range = dataMax_[j] - dataMin_[j];
                float unscaled = (input[i * nFeatures_ + j] - featureMin_) / (featureMax_ - featureMin_);
                output[i * nFeatures_ + j] = unscaled * range + dataMin_[j];
            }
        }
    }
    
private:
    std::vector<float> dataMin_;
    std::vector<float> dataMax_;
    float featureMin_, featureMax_;
    int nFeatures_ = 0;
    bool fitted_;
};

/**
 * @brief Normalizer - normalize samples to unit norm
 */
class Normalizer {
public:
    enum Norm { L1, L2, MAX };
    
    Normalizer(Norm norm = L2) : norm_(norm) {}
    
    void transform(const float* input, float* output, int nSamples, int nFeatures) const {
        for (int i = 0; i < nSamples; ++i) {
            float norm = computeNorm(input + i * nFeatures, nFeatures);
            if (norm < 1e-10f) norm = 1.0f;
            
            for (int j = 0; j < nFeatures; ++j) {
                output[i * nFeatures + j] = input[i * nFeatures + j] / norm;
            }
        }
    }
    
private:
    float computeNorm(const float* sample, int nFeatures) const {
        float norm = 0.0f;
        switch (norm_) {
            case L1:
                for (int j = 0; j < nFeatures; ++j) {
                    norm += std::abs(sample[j]);
                }
                break;
            case L2:
                for (int j = 0; j < nFeatures; ++j) {
                    norm += sample[j] * sample[j];
                }
                norm = std::sqrt(norm);
                break;
            case MAX:
                for (int j = 0; j < nFeatures; ++j) {
                    norm = std::max(norm, std::abs(sample[j]));
                }
                break;
        }
        return norm;
    }
    
    Norm norm_;
};

/**
 * @brief Label encoder - encode categorical labels to integers
 */
class LabelEncoder {
public:
    void fit(const int* labels, int nSamples) {
        labelToIdx_.clear();
        idxToLabel_.clear();
        
        for (int i = 0; i < nSamples; ++i) {
            if (labelToIdx_.find(labels[i]) == labelToIdx_.end()) {
                int idx = static_cast<int>(labelToIdx_.size());
                labelToIdx_[labels[i]] = idx;
                idxToLabel_[idx] = labels[i];
            }
        }
    }
    
    void transform(const int* input, int* output, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            auto it = labelToIdx_.find(input[i]);
            output[i] = (it != labelToIdx_.end()) ? it->second : -1;
        }
    }
    
    void inverseTransform(const int* input, int* output, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            auto it = idxToLabel_.find(input[i]);
            output[i] = (it != idxToLabel_.end()) ? it->second : -1;
        }
    }
    
    int numClasses() const { return static_cast<int>(labelToIdx_.size()); }
    
private:
    std::map<int, int> labelToIdx_;
    std::map<int, int> idxToLabel_;
};

/**
 * @brief One-hot encoder
 */
class OneHotEncoder {
public:
    void fit(const int* labels, int nSamples) {
        encoder_.fit(labels, nSamples);
    }
    
    void transform(const int* input, float* output, int nSamples) const {
        int nClasses = encoder_.numClasses();
        std::fill(output, output + nSamples * nClasses, 0.0f);
        
        std::vector<int> encoded(nSamples);
        encoder_.transform(input, encoded.data(), nSamples);
        
        for (int i = 0; i < nSamples; ++i) {
            if (encoded[i] >= 0 && encoded[i] < nClasses) {
                output[i * nClasses + encoded[i]] = 1.0f;
            }
        }
    }
    
    int numClasses() const { return encoder_.numClasses(); }
    
private:
    LabelEncoder encoder_;
};

/**
 * @brief Binarizer - binarize data based on threshold
 */
class Binarizer {
public:
    Binarizer(float threshold = 0.0f) : threshold_(threshold) {}
    
    void transform(const float* input, float* output, int size) const {
        for (int i = 0; i < size; ++i) {
            output[i] = (input[i] > threshold_) ? 1.0f : 0.0f;
        }
    }
    
private:
    float threshold_;
};

/**
 * @brief Polynomial feature generator
 */
class PolynomialFeatures {
public:
    PolynomialFeatures(int degree = 2, bool includeBias = true)
        : degree_(degree), includeBias_(includeBias) {}
    
    int numOutputFeatures(int nFeatures) const {
        // Compute number of polynomial features
        int n = includeBias_ ? 1 : 0;
        for (int d = 1; d <= degree_; ++d) {
            n += binomial(nFeatures + d - 1, d);
        }
        return n;
    }
    
    void transform(const float* input, float* output, int nSamples, int nFeatures) const {
        int nOutput = numOutputFeatures(nFeatures);
        
        for (int i = 0; i < nSamples; ++i) {
            int outIdx = 0;
            
            // Bias term
            if (includeBias_) {
                output[i * nOutput + outIdx++] = 1.0f;
            }
            
            // Degree 1 (original features)
            for (int j = 0; j < nFeatures; ++j) {
                output[i * nOutput + outIdx++] = input[i * nFeatures + j];
            }
            
            // Higher degree terms (simplified: just powers)
            for (int d = 2; d <= degree_; ++d) {
                for (int j = 0; j < nFeatures; ++j) {
                    float val = std::pow(input[i * nFeatures + j], d);
                    output[i * nOutput + outIdx++] = val;
                }
            }
        }
    }
    
private:
    int binomial(int n, int k) const {
        if (k > n) return 0;
        if (k == 0 || k == n) return 1;
        return binomial(n - 1, k - 1) + binomial(n - 1, k);
    }
    
    int degree_;
    bool includeBias_;
};

/**
 * @brief Imputer for handling missing values
 */
class SimpleImputer {
public:
    enum Strategy { MEAN, MEDIAN, MOST_FREQUENT, CONSTANT };
    
    SimpleImputer(Strategy strategy = MEAN, float fillValue = 0.0f, float missingValue = NAN)
        : strategy_(strategy), fillValue_(fillValue), missingValue_(missingValue) {}
    
    void fit(const float* data, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        fillValues_.resize(nFeatures);
        
        for (int j = 0; j < nFeatures; ++j) {
            std::vector<float> valid;
            for (int i = 0; i < nSamples; ++i) {
                float val = data[i * nFeatures + j];
                if (!std::isnan(val) && val != missingValue_) {
                    valid.push_back(val);
                }
            }
            
            if (valid.empty()) {
                fillValues_[j] = fillValue_;
                continue;
            }
            
            switch (strategy_) {
                case MEAN: {
                    float sum = std::accumulate(valid.begin(), valid.end(), 0.0f);
                    fillValues_[j] = sum / valid.size();
                    break;
                }
                case MEDIAN: {
                    std::sort(valid.begin(), valid.end());
                    size_t mid = valid.size() / 2;
                    fillValues_[j] = (valid.size() % 2 == 0) ?
                        (valid[mid - 1] + valid[mid]) / 2.0f : valid[mid];
                    break;
                }
                case MOST_FREQUENT: {
                    std::map<float, int> counts;
                    for (float v : valid) counts[v]++;
                    int maxCount = 0;
                    for (const auto& [v, c] : counts) {
                        if (c > maxCount) {
                            maxCount = c;
                            fillValues_[j] = v;
                        }
                    }
                    break;
                }
                case CONSTANT:
                    fillValues_[j] = fillValue_;
                    break;
            }
        }
    }
    
    void transform(const float* input, float* output, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nFeatures_; ++j) {
                float val = input[i * nFeatures_ + j];
                if (std::isnan(val) || val == missingValue_) {
                    output[i * nFeatures_ + j] = fillValues_[j];
                } else {
                    output[i * nFeatures_ + j] = val;
                }
            }
        }
    }
    
private:
    Strategy strategy_;
    float fillValue_;
    float missingValue_;
    std::vector<float> fillValues_;
    int nFeatures_ = 0;
};

/**
 * @brief Split data into train and test sets
 */
inline void trainTestSplit(
    const float* X, const int* y,
    int nSamples, int nFeatures,
    float* XTrain, float* XTest,
    int* yTrain, int* yTest,
    int& nTrain, int& nTest,
    float testSize = 0.2f,
    bool shuffle = true,
    unsigned int seed = 42
) {
    nTest = static_cast<int>(nSamples * testSize);
    nTrain = nSamples - nTest;
    
    std::vector<int> indices(nSamples);
    std::iota(indices.begin(), indices.end(), 0);
    
    if (shuffle) {
        // Simple LCG shuffle
        unsigned int state = seed;
        for (int i = nSamples - 1; i > 0; --i) {
            state = state * 1103515245 + 12345;
            int j = (state / 65536) % (i + 1);
            std::swap(indices[i], indices[j]);
        }
    }
    
    for (int i = 0; i < nTrain; ++i) {
        int idx = indices[i];
        for (int j = 0; j < nFeatures; ++j) {
            XTrain[i * nFeatures + j] = X[idx * nFeatures + j];
        }
        yTrain[i] = y[idx];
    }
    
    for (int i = 0; i < nTest; ++i) {
        int idx = indices[nTrain + i];
        for (int j = 0; j < nFeatures; ++j) {
            XTest[i * nFeatures + j] = X[idx * nFeatures + j];
        }
        yTest[i] = y[idx];
    }
}

} // namespace ml
} // namespace neurova

#endif // NEUROVA_ML_PREPROCESSING_HPP
