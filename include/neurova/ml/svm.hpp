// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file svm.hpp
 * @brief Support Vector Machine implementation
 */

#ifndef NEUROVA_ML_SVM_HPP
#define NEUROVA_ML_SVM_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <functional>

namespace neurova {
namespace ml {

/**
 * @brief Kernel functions
 */
namespace kernel {

/**
 * @brief Linear kernel: K(x, y) = x · y
 */
inline float linear(const float* x, const float* y, int n, float gamma = 1.0f, float coef0 = 0.0f, int degree = 3) {
    (void)gamma; (void)coef0; (void)degree;
    float dot = 0.0f;
    for (int i = 0; i < n; ++i) {
        dot += x[i] * y[i];
    }
    return dot;
}

/**
 * @brief Polynomial kernel: K(x, y) = (gamma * x · y + coef0)^degree
 */
inline float polynomial(const float* x, const float* y, int n, float gamma = 1.0f, float coef0 = 0.0f, int degree = 3) {
    float dot = 0.0f;
    for (int i = 0; i < n; ++i) {
        dot += x[i] * y[i];
    }
    return std::pow(gamma * dot + coef0, degree);
}

/**
 * @brief RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
 */
inline float rbf(const float* x, const float* y, int n, float gamma = 1.0f, float coef0 = 0.0f, int degree = 3) {
    (void)coef0; (void)degree;
    float dist = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = x[i] - y[i];
        dist += diff * diff;
    }
    return std::exp(-gamma * dist);
}

/**
 * @brief Sigmoid kernel: K(x, y) = tanh(gamma * x · y + coef0)
 */
inline float sigmoid(const float* x, const float* y, int n, float gamma = 1.0f, float coef0 = 0.0f, int degree = 3) {
    (void)degree;
    float dot = 0.0f;
    for (int i = 0; i < n; ++i) {
        dot += x[i] * y[i];
    }
    return std::tanh(gamma * dot + coef0);
}

} // namespace kernel

/**
 * @brief Support Vector Classifier
 */
class SVC {
public:
    enum KernelType { LINEAR, POLY, RBF, SIGMOID };
    
    SVC(float C = 1.0f, KernelType kernel = RBF, float gamma = 0.1f, float coef0 = 0.0f, int degree = 3, float tol = 1e-3f, int maxIter = 1000)
        : C_(C), kernelType_(kernel), gamma_(gamma), coef0_(coef0), degree_(degree), tol_(tol), maxIter_(maxIter) {}
    
    void fit(const float* X, const int* y, int nSamples, int nFeatures) {
        nSamples_ = nSamples;
        nFeatures_ = nFeatures;
        
        // Store training data
        X_.assign(X, X + nSamples * nFeatures);
        y_.resize(nSamples);
        for (int i = 0; i < nSamples; ++i) {
            y_[i] = (y[i] > 0) ? 1.0f : -1.0f;
        }
        
        // Initialize alphas and bias
        alphas_.resize(nSamples, 0.0f);
        b_ = 0.0f;
        
        // Precompute kernel matrix
        kernelMatrix_.resize(nSamples * nSamples);
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nSamples; ++j) {
                kernelMatrix_[i * nSamples + j] = computeKernel(X + i * nFeatures, X + j * nFeatures);
            }
        }
        
        // SMO algorithm
        int iter = 0;
        int passes = 0;
        
        while (passes < 5 && iter < maxIter_) {
            int numChangedAlphas = 0;
            
            for (int i = 0; i < nSamples; ++i) {
                float Ei = computeError(i);
                
                if ((y_[i] * Ei < -tol_ && alphas_[i] < C_) ||
                    (y_[i] * Ei > tol_ && alphas_[i] > 0)) {
                    
                    // Select j != i randomly
                    int j = (i + 1 + (iter % (nSamples - 1))) % nSamples;
                    
                    float Ej = computeError(j);
                    
                    float alphaIOld = alphas_[i];
                    float alphaJOld = alphas_[j];
                    
                    // Compute L and H
                    float L, H;
                    if (y_[i] != y_[j]) {
                        L = std::max(0.0f, alphas_[j] - alphas_[i]);
                        H = std::min(C_, C_ + alphas_[j] - alphas_[i]);
                    } else {
                        L = std::max(0.0f, alphas_[i] + alphas_[j] - C_);
                        H = std::min(C_, alphas_[i] + alphas_[j]);
                    }
                    
                    if (L >= H) continue;
                    
                    // Compute eta
                    float eta = 2 * kernelMatrix_[i * nSamples + j] - 
                               kernelMatrix_[i * nSamples + i] - 
                               kernelMatrix_[j * nSamples + j];
                    
                    if (eta >= 0) continue;
                    
                    // Update alpha_j
                    alphas_[j] -= y_[j] * (Ei - Ej) / eta;
                    alphas_[j] = std::max(L, std::min(H, alphas_[j]));
                    
                    if (std::abs(alphas_[j] - alphaJOld) < 1e-5f) continue;
                    
                    // Update alpha_i
                    alphas_[i] += y_[i] * y_[j] * (alphaJOld - alphas_[j]);
                    
                    // Update bias
                    float b1 = b_ - Ei - y_[i] * (alphas_[i] - alphaIOld) * kernelMatrix_[i * nSamples + i] -
                              y_[j] * (alphas_[j] - alphaJOld) * kernelMatrix_[i * nSamples + j];
                    float b2 = b_ - Ej - y_[i] * (alphas_[i] - alphaIOld) * kernelMatrix_[i * nSamples + j] -
                              y_[j] * (alphas_[j] - alphaJOld) * kernelMatrix_[j * nSamples + j];
                    
                    if (0 < alphas_[i] && alphas_[i] < C_) {
                        b_ = b1;
                    } else if (0 < alphas_[j] && alphas_[j] < C_) {
                        b_ = b2;
                    } else {
                        b_ = (b1 + b2) / 2;
                    }
                    
                    numChangedAlphas++;
                }
            }
            
            if (numChangedAlphas == 0) {
                passes++;
            } else {
                passes = 0;
            }
            
            iter++;
        }
        
        // Extract support vectors
        supportVectors_.clear();
        supportVectorIndices_.clear();
        supportVectorAlphas_.clear();
        supportVectorLabels_.clear();
        
        for (int i = 0; i < nSamples; ++i) {
            if (alphas_[i] > 1e-8f) {
                supportVectorIndices_.push_back(i);
                supportVectorAlphas_.push_back(alphas_[i]);
                supportVectorLabels_.push_back(y_[i]);
                std::vector<float> sv(X + i * nFeatures, X + (i + 1) * nFeatures);
                supportVectors_.push_back(sv);
            }
        }
    }
    
    int predict(const float* sample) const {
        float decision = decisionFunction(sample);
        return (decision >= 0) ? 1 : -1;
    }
    
    void predict(const float* X, int* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
    float decisionFunction(const float* sample) const {
        float sum = b_;
        for (size_t i = 0; i < supportVectors_.size(); ++i) {
            sum += supportVectorAlphas_[i] * supportVectorLabels_[i] * 
                   computeKernel(supportVectors_[i].data(), sample);
        }
        return sum;
    }
    
    int numSupportVectors() const { return static_cast<int>(supportVectors_.size()); }
    
private:
    float computeKernel(const float* x, const float* y) const {
        switch (kernelType_) {
            case LINEAR: return kernel::linear(x, y, nFeatures_, gamma_, coef0_, degree_);
            case POLY: return kernel::polynomial(x, y, nFeatures_, gamma_, coef0_, degree_);
            case RBF: return kernel::rbf(x, y, nFeatures_, gamma_, coef0_, degree_);
            case SIGMOID: return kernel::sigmoid(x, y, nFeatures_, gamma_, coef0_, degree_);
            default: return kernel::rbf(x, y, nFeatures_, gamma_, coef0_, degree_);
        }
    }
    
    float computeError(int i) const {
        float sum = b_;
        for (int j = 0; j < nSamples_; ++j) {
            if (alphas_[j] > 0) {
                sum += alphas_[j] * y_[j] * kernelMatrix_[i * nSamples_ + j];
            }
        }
        return sum - y_[i];
    }
    
    float C_;
    KernelType kernelType_;
    float gamma_;
    float coef0_;
    int degree_;
    float tol_;
    int maxIter_;
    
    std::vector<float> X_;
    std::vector<float> y_;
    std::vector<float> alphas_;
    std::vector<float> kernelMatrix_;
    float b_ = 0.0f;
    
    std::vector<std::vector<float>> supportVectors_;
    std::vector<int> supportVectorIndices_;
    std::vector<float> supportVectorAlphas_;
    std::vector<float> supportVectorLabels_;
    
    int nSamples_ = 0;
    int nFeatures_ = 0;
};

/**
 * @brief Support Vector Regression
 */
class SVR {
public:
    enum KernelType { LINEAR, POLY, RBF, SIGMOID };
    
    SVR(float C = 1.0f, float epsilon = 0.1f, KernelType kernel = RBF, float gamma = 0.1f)
        : C_(C), epsilon_(epsilon), kernelType_(kernel), gamma_(gamma) {}
    
    void fit(const float* X, const float* y, int nSamples, int nFeatures) {
        nSamples_ = nSamples;
        nFeatures_ = nFeatures;
        
        X_.assign(X, X + nSamples * nFeatures);
        y_.assign(y, y + nSamples);
        
        // Simplified: gradient descent on dual
        alphas_.resize(nSamples, 0.0f);
        alphasStar_.resize(nSamples, 0.0f);
        b_ = 0.0f;
        
        // Precompute kernel matrix
        kernelMatrix_.resize(nSamples * nSamples);
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nSamples; ++j) {
                kernelMatrix_[i * nSamples + j] = computeKernel(X + i * nFeatures, X + j * nFeatures);
            }
        }
        
        // Simplified SMO for SVR
        float learningRate = 0.01f;
        int maxIter = 1000;
        
        for (int iter = 0; iter < maxIter; ++iter) {
            for (int i = 0; i < nSamples; ++i) {
                float pred = predictInternal(i);
                float error = pred - y_[i];
                
                // Update alphas
                if (error > epsilon_) {
                    alphas_[i] = std::max(0.0f, std::min(C_, alphas_[i] - learningRate * error));
                } else if (error < -epsilon_) {
                    alphasStar_[i] = std::max(0.0f, std::min(C_, alphasStar_[i] + learningRate * error));
                }
            }
            
            // Update bias
            float sumAlpha = 0.0f;
            int count = 0;
            for (int i = 0; i < nSamples; ++i) {
                if ((alphas_[i] > 0 && alphas_[i] < C_) || (alphasStar_[i] > 0 && alphasStar_[i] < C_)) {
                    float pred = 0.0f;
                    for (int j = 0; j < nSamples; ++j) {
                        pred += (alphas_[j] - alphasStar_[j]) * kernelMatrix_[i * nSamples + j];
                    }
                    sumAlpha += y_[i] - pred;
                    count++;
                }
            }
            if (count > 0) b_ = sumAlpha / count;
        }
    }
    
    float predict(const float* sample) const {
        float sum = b_;
        for (int i = 0; i < nSamples_; ++i) {
            float k = computeKernel(X_.data() + i * nFeatures_, sample);
            sum += (alphas_[i] - alphasStar_[i]) * k;
        }
        return sum;
    }
    
    void predict(const float* X, float* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
private:
    float computeKernel(const float* x, const float* y) const {
        switch (kernelType_) {
            case LINEAR: return kernel::linear(x, y, nFeatures_, gamma_, 0.0f, 3);
            case RBF: return kernel::rbf(x, y, nFeatures_, gamma_, 0.0f, 3);
            default: return kernel::rbf(x, y, nFeatures_, gamma_, 0.0f, 3);
        }
    }
    
    float predictInternal(int idx) const {
        float sum = b_;
        for (int j = 0; j < nSamples_; ++j) {
            sum += (alphas_[j] - alphasStar_[j]) * kernelMatrix_[idx * nSamples_ + j];
        }
        return sum;
    }
    
    float C_;
    float epsilon_;
    KernelType kernelType_;
    float gamma_;
    
    std::vector<float> X_;
    std::vector<float> y_;
    std::vector<float> alphas_;
    std::vector<float> alphasStar_;
    std::vector<float> kernelMatrix_;
    float b_ = 0.0f;
    
    int nSamples_ = 0;
    int nFeatures_ = 0;
};

/**
 * @brief One-class SVM for outlier detection
 */
class OneClassSVM {
public:
    OneClassSVM(float nu = 0.5f, float gamma = 0.1f)
        : nu_(nu), gamma_(gamma) {}
    
    void fit(const float* X, int nSamples, int nFeatures) {
        nSamples_ = nSamples;
        nFeatures_ = nFeatures;
        
        X_.assign(X, X + nSamples * nFeatures);
        alphas_.resize(nSamples, 0.0f);
        
        // Initialize alphas uniformly
        float initAlpha = nu_ / nSamples;
        for (int i = 0; i < nSamples; ++i) {
            alphas_[i] = initAlpha;
        }
        
        // Simplified optimization
        rho_ = 0.0f;
        for (int i = 0; i < nSamples; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < nSamples; ++j) {
                sum += alphas_[j] * kernel::rbf(X + i * nFeatures, X + j * nFeatures, nFeatures, gamma_);
            }
            rho_ += sum;
        }
        rho_ /= nSamples;
    }
    
    int predict(const float* sample) const {
        return (decisionFunction(sample) >= 0) ? 1 : -1;
    }
    
    float decisionFunction(const float* sample) const {
        float sum = 0.0f;
        for (int i = 0; i < nSamples_; ++i) {
            sum += alphas_[i] * kernel::rbf(X_.data() + i * nFeatures_, sample, nFeatures_, gamma_);
        }
        return sum - rho_;
    }
    
private:
    float nu_;
    float gamma_;
    
    std::vector<float> X_;
    std::vector<float> alphas_;
    float rho_ = 0.0f;
    
    int nSamples_ = 0;
    int nFeatures_ = 0;
};

} // namespace ml
} // namespace neurova

#endif // NEUROVA_ML_SVM_HPP
