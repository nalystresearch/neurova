// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file regression.hpp
 * @brief Regression algorithms
 */

#ifndef NEUROVA_ML_REGRESSION_HPP
#define NEUROVA_ML_REGRESSION_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace neurova {
namespace ml {

/**
 * @brief Linear Regression
 */
class LinearRegression {
public:
    LinearRegression(bool fitIntercept = true) : fitIntercept_(fitIntercept) {}
    
    void fit(const float* X, const float* y, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        int nCoeffs = fitIntercept_ ? nFeatures + 1 : nFeatures;
        coefficients_.resize(nCoeffs, 0.0f);
        
        // Normal equation: w = (X^T X)^(-1) X^T y
        // For simplicity, use gradient descent
        float learningRate = 0.01f;
        int maxIter = 1000;
        
        for (int iter = 0; iter < maxIter; ++iter) {
            std::vector<float> grad(nCoeffs, 0.0f);
            
            for (int i = 0; i < nSamples; ++i) {
                float pred = predictSingle(X + i * nFeatures);
                float error = pred - y[i];
                
                int offset = 0;
                if (fitIntercept_) {
                    grad[0] += error;
                    offset = 1;
                }
                
                for (int j = 0; j < nFeatures; ++j) {
                    grad[offset + j] += error * X[i * nFeatures + j];
                }
            }
            
            // Update
            for (int j = 0; j < nCoeffs; ++j) {
                coefficients_[j] -= learningRate * grad[j] / nSamples;
            }
        }
        
        if (fitIntercept_) {
            intercept_ = coefficients_[0];
            coef_.assign(coefficients_.begin() + 1, coefficients_.end());
        } else {
            intercept_ = 0.0f;
            coef_ = coefficients_;
        }
    }
    
    float predict(const float* sample) const {
        return predictSingle(sample);
    }
    
    void predict(const float* X, float* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
    float score(const float* X, const float* y, int nSamples) const {
        // R^2 score
        float ssRes = 0.0f, ssTot = 0.0f;
        float yMean = 0.0f;
        
        for (int i = 0; i < nSamples; ++i) yMean += y[i];
        yMean /= nSamples;
        
        for (int i = 0; i < nSamples; ++i) {
            float pred = predict(X + i * nFeatures_);
            ssRes += (y[i] - pred) * (y[i] - pred);
            ssTot += (y[i] - yMean) * (y[i] - yMean);
        }
        
        return 1.0f - ssRes / ssTot;
    }
    
    const std::vector<float>& coef() const { return coef_; }
    float intercept() const { return intercept_; }
    
private:
    float predictSingle(const float* sample) const {
        float pred = 0.0f;
        int offset = 0;
        
        if (fitIntercept_) {
            pred = coefficients_[0];
            offset = 1;
        }
        
        for (int j = 0; j < nFeatures_; ++j) {
            pred += coefficients_[offset + j] * sample[j];
        }
        
        return pred;
    }
    
    bool fitIntercept_;
    std::vector<float> coefficients_;
    std::vector<float> coef_;
    float intercept_ = 0.0f;
    int nFeatures_ = 0;
};

/**
 * @brief Ridge Regression (L2 regularized linear regression)
 */
class Ridge {
public:
    Ridge(float alpha = 1.0f, bool fitIntercept = true)
        : alpha_(alpha), fitIntercept_(fitIntercept) {}
    
    void fit(const float* X, const float* y, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        int nCoeffs = fitIntercept_ ? nFeatures + 1 : nFeatures;
        coefficients_.resize(nCoeffs, 0.0f);
        
        float learningRate = 0.01f;
        int maxIter = 1000;
        
        for (int iter = 0; iter < maxIter; ++iter) {
            std::vector<float> grad(nCoeffs, 0.0f);
            
            for (int i = 0; i < nSamples; ++i) {
                float pred = predictSingle(X + i * nFeatures);
                float error = pred - y[i];
                
                int offset = 0;
                if (fitIntercept_) {
                    grad[0] += error;
                    offset = 1;
                }
                
                for (int j = 0; j < nFeatures; ++j) {
                    grad[offset + j] += error * X[i * nFeatures + j];
                }
            }
            
            // Update with regularization
            int offset = fitIntercept_ ? 1 : 0;
            for (int j = 0; j < nCoeffs; ++j) {
                float reg = (j >= offset) ? alpha_ * coefficients_[j] : 0.0f;
                coefficients_[j] -= learningRate * (grad[j] / nSamples + reg);
            }
        }
        
        if (fitIntercept_) {
            intercept_ = coefficients_[0];
            coef_.assign(coefficients_.begin() + 1, coefficients_.end());
        } else {
            intercept_ = 0.0f;
            coef_ = coefficients_;
        }
    }
    
    float predict(const float* sample) const {
        return predictSingle(sample);
    }
    
    void predict(const float* X, float* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
private:
    float predictSingle(const float* sample) const {
        float pred = 0.0f;
        int offset = 0;
        
        if (fitIntercept_) {
            pred = coefficients_[0];
            offset = 1;
        }
        
        for (int j = 0; j < nFeatures_; ++j) {
            pred += coefficients_[offset + j] * sample[j];
        }
        
        return pred;
    }
    
    float alpha_;
    bool fitIntercept_;
    std::vector<float> coefficients_;
    std::vector<float> coef_;
    float intercept_ = 0.0f;
    int nFeatures_ = 0;
};

/**
 * @brief Lasso Regression (L1 regularized linear regression)
 */
class Lasso {
public:
    Lasso(float alpha = 1.0f, bool fitIntercept = true, int maxIter = 1000)
        : alpha_(alpha), fitIntercept_(fitIntercept), maxIter_(maxIter) {}
    
    void fit(const float* X, const float* y, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        int nCoeffs = fitIntercept_ ? nFeatures + 1 : nFeatures;
        coefficients_.resize(nCoeffs, 0.0f);
        
        float learningRate = 0.01f;
        
        for (int iter = 0; iter < maxIter_; ++iter) {
            std::vector<float> grad(nCoeffs, 0.0f);
            
            for (int i = 0; i < nSamples; ++i) {
                float pred = predictSingle(X + i * nFeatures);
                float error = pred - y[i];
                
                int offset = 0;
                if (fitIntercept_) {
                    grad[0] += error;
                    offset = 1;
                }
                
                for (int j = 0; j < nFeatures; ++j) {
                    grad[offset + j] += error * X[i * nFeatures + j];
                }
            }
            
            // Proximal gradient update for L1
            int offset = fitIntercept_ ? 1 : 0;
            for (int j = 0; j < nCoeffs; ++j) {
                float update = coefficients_[j] - learningRate * grad[j] / nSamples;
                
                if (j >= offset) {
                    // Soft thresholding
                    float threshold = alpha_ * learningRate;
                    if (update > threshold) {
                        coefficients_[j] = update - threshold;
                    } else if (update < -threshold) {
                        coefficients_[j] = update + threshold;
                    } else {
                        coefficients_[j] = 0.0f;
                    }
                } else {
                    coefficients_[j] = update;
                }
            }
        }
        
        if (fitIntercept_) {
            intercept_ = coefficients_[0];
            coef_.assign(coefficients_.begin() + 1, coefficients_.end());
        } else {
            intercept_ = 0.0f;
            coef_ = coefficients_;
        }
    }
    
    float predict(const float* sample) const {
        return predictSingle(sample);
    }
    
    void predict(const float* X, float* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
private:
    float predictSingle(const float* sample) const {
        float pred = 0.0f;
        int offset = 0;
        
        if (fitIntercept_) {
            pred = coefficients_[0];
            offset = 1;
        }
        
        for (int j = 0; j < nFeatures_; ++j) {
            pred += coefficients_[offset + j] * sample[j];
        }
        
        return pred;
    }
    
    float alpha_;
    bool fitIntercept_;
    int maxIter_;
    std::vector<float> coefficients_;
    std::vector<float> coef_;
    float intercept_ = 0.0f;
    int nFeatures_ = 0;
};

/**
 * @brief ElasticNet (L1 + L2 regularized linear regression)
 */
class ElasticNet {
public:
    ElasticNet(float alpha = 1.0f, float l1Ratio = 0.5f, bool fitIntercept = true)
        : alpha_(alpha), l1Ratio_(l1Ratio), fitIntercept_(fitIntercept) {}
    
    void fit(const float* X, const float* y, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        int nCoeffs = fitIntercept_ ? nFeatures + 1 : nFeatures;
        coefficients_.resize(nCoeffs, 0.0f);
        
        float learningRate = 0.01f;
        int maxIter = 1000;
        
        float l1Weight = alpha_ * l1Ratio_;
        float l2Weight = alpha_ * (1.0f - l1Ratio_);
        
        for (int iter = 0; iter < maxIter; ++iter) {
            std::vector<float> grad(nCoeffs, 0.0f);
            
            for (int i = 0; i < nSamples; ++i) {
                float pred = predictSingle(X + i * nFeatures);
                float error = pred - y[i];
                
                int offset = 0;
                if (fitIntercept_) {
                    grad[0] += error;
                    offset = 1;
                }
                
                for (int j = 0; j < nFeatures; ++j) {
                    grad[offset + j] += error * X[i * nFeatures + j];
                }
            }
            
            // Update with both L1 and L2
            int offset = fitIntercept_ ? 1 : 0;
            for (int j = 0; j < nCoeffs; ++j) {
                // L2 gradient
                float l2Grad = (j >= offset) ? l2Weight * coefficients_[j] : 0.0f;
                float update = coefficients_[j] - learningRate * (grad[j] / nSamples + l2Grad);
                
                // L1 soft thresholding
                if (j >= offset) {
                    float threshold = l1Weight * learningRate;
                    if (update > threshold) {
                        coefficients_[j] = update - threshold;
                    } else if (update < -threshold) {
                        coefficients_[j] = update + threshold;
                    } else {
                        coefficients_[j] = 0.0f;
                    }
                } else {
                    coefficients_[j] = update;
                }
            }
        }
        
        if (fitIntercept_) {
            intercept_ = coefficients_[0];
            coef_.assign(coefficients_.begin() + 1, coefficients_.end());
        } else {
            intercept_ = 0.0f;
            coef_ = coefficients_;
        }
    }
    
    float predict(const float* sample) const {
        return predictSingle(sample);
    }
    
    void predict(const float* X, float* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
private:
    float predictSingle(const float* sample) const {
        float pred = 0.0f;
        int offset = 0;
        
        if (fitIntercept_) {
            pred = coefficients_[0];
            offset = 1;
        }
        
        for (int j = 0; j < nFeatures_; ++j) {
            pred += coefficients_[offset + j] * sample[j];
        }
        
        return pred;
    }
    
    float alpha_;
    float l1Ratio_;
    bool fitIntercept_;
    std::vector<float> coefficients_;
    std::vector<float> coef_;
    float intercept_ = 0.0f;
    int nFeatures_ = 0;
};

/**
 * @brief Polynomial Regression
 */
class PolynomialRegression {
public:
    PolynomialRegression(int degree = 2) : degree_(degree) {}
    
    void fit(const float* X, const float* y, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        
        // Generate polynomial features
        int nPolyFeatures = computePolyFeatureCount(nFeatures, degree_);
        std::vector<float> polyX(nSamples * nPolyFeatures);
        
        generatePolyFeatures(X, polyX.data(), nSamples, nFeatures);
        
        // Fit linear regression on polynomial features
        regressor_.fit(polyX.data(), y, nSamples, nPolyFeatures);
    }
    
    float predict(const float* sample) const {
        std::vector<float> polySample(computePolyFeatureCount(nFeatures_, degree_));
        generatePolyFeaturesSingle(sample, polySample.data(), nFeatures_);
        return regressor_.predict(polySample.data());
    }
    
    void predict(const float* X, float* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
private:
    int computePolyFeatureCount(int nFeatures, int degree) const {
        // 1 (bias) + nFeatures * degree
        return 1 + nFeatures * degree;
    }
    
    void generatePolyFeatures(const float* X, float* polyX, int nSamples, int nFeatures) const {
        int nPoly = computePolyFeatureCount(nFeatures, degree_);
        
        for (int i = 0; i < nSamples; ++i) {
            generatePolyFeaturesSingle(X + i * nFeatures, polyX + i * nPoly, nFeatures);
        }
    }
    
    void generatePolyFeaturesSingle(const float* sample, float* polySample, int nFeatures) const {
        int idx = 0;
        polySample[idx++] = 1.0f;  // Bias
        
        for (int d = 1; d <= degree_; ++d) {
            for (int j = 0; j < nFeatures; ++j) {
                polySample[idx++] = std::pow(sample[j], d);
            }
        }
    }
    
    int degree_;
    int nFeatures_ = 0;
    LinearRegression regressor_{false};  // No intercept since we include bias
};

/**
 * @brief K-Nearest Neighbors Regressor
 */
class KNeighborsRegressor {
public:
    KNeighborsRegressor(int k = 5, const std::string& weights = "uniform")
        : k_(k), weights_(weights) {}
    
    void fit(const float* X, const float* y, int nSamples, int nFeatures) {
        nSamples_ = nSamples;
        nFeatures_ = nFeatures;
        X_.assign(X, X + nSamples * nFeatures);
        y_.assign(y, y + nSamples);
    }
    
    float predict(const float* sample) const {
        // Find k nearest neighbors
        std::vector<std::pair<float, float>> distances;  // (dist, y)
        
        for (int i = 0; i < nSamples_; ++i) {
            float dist = 0.0f;
            for (int j = 0; j < nFeatures_; ++j) {
                float diff = sample[j] - X_[i * nFeatures_ + j];
                dist += diff * diff;
            }
            distances.push_back({std::sqrt(dist), y_[i]});
        }
        
        std::partial_sort(distances.begin(), distances.begin() + k_, distances.end());
        
        // Weighted average
        float sum = 0.0f, weightSum = 0.0f;
        for (int i = 0; i < k_; ++i) {
            float weight = 1.0f;
            if (weights_ == "distance" && distances[i].first > 1e-10f) {
                weight = 1.0f / distances[i].first;
            }
            sum += weight * distances[i].second;
            weightSum += weight;
        }
        
        return sum / weightSum;
    }
    
    void predict(const float* X, float* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
private:
    int k_;
    std::string weights_;
    std::vector<float> X_;
    std::vector<float> y_;
    int nSamples_ = 0;
    int nFeatures_ = 0;
};

/**
 * @brief Decision Tree Regressor
 */
class DecisionTreeRegressor {
public:
    DecisionTreeRegressor(int maxDepth = 10, int minSamplesSplit = 2, int minSamplesLeaf = 1)
        : maxDepth_(maxDepth), minSamplesSplit_(minSamplesSplit), minSamplesLeaf_(minSamplesLeaf) {}
    
    void fit(const float* X, const float* y, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        std::vector<int> indices(nSamples);
        std::iota(indices.begin(), indices.end(), 0);
        root_ = buildTree(X, y, indices, 0);
    }
    
    float predict(const float* sample) const {
        const Node* node = root_.get();
        while (node && !node->isLeaf) {
            if (sample[node->featureIdx] <= node->threshold) {
                node = node->left.get();
            } else {
                node = node->right.get();
            }
        }
        return node ? node->prediction : 0.0f;
    }
    
    void predict(const float* X, float* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
private:
    struct Node {
        bool isLeaf = false;
        int featureIdx = -1;
        float threshold = 0.0f;
        float prediction = 0.0f;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };
    
    std::unique_ptr<Node> buildTree(const float* X, const float* y,
                                    const std::vector<int>& indices, int depth) {
        auto node = std::make_unique<Node>();
        
        if (indices.empty()) {
            node->isLeaf = true;
            return node;
        }
        
        // Compute mean prediction
        float mean = 0.0f;
        for (int idx : indices) mean += y[idx];
        mean /= indices.size();
        
        // Check stopping criteria
        if (depth >= maxDepth_ || static_cast<int>(indices.size()) < minSamplesSplit_) {
            node->isLeaf = true;
            node->prediction = mean;
            return node;
        }
        
        // Find best split (minimize MSE)
        float bestMSE = std::numeric_limits<float>::max();
        int bestFeature = 0;
        float bestThreshold = 0.0f;
        
        for (int f = 0; f < nFeatures_; ++f) {
            std::vector<float> values;
            for (int idx : indices) {
                values.push_back(X[idx * nFeatures_ + f]);
            }
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
            
            for (size_t i = 0; i < values.size() - 1; ++i) {
                float threshold = (values[i] + values[i + 1]) / 2.0f;
                
                float leftSum = 0.0f, rightSum = 0.0f;
                int leftCount = 0, rightCount = 0;
                
                for (int idx : indices) {
                    if (X[idx * nFeatures_ + f] <= threshold) {
                        leftSum += y[idx];
                        leftCount++;
                    } else {
                        rightSum += y[idx];
                        rightCount++;
                    }
                }
                
                if (leftCount < minSamplesLeaf_ || rightCount < minSamplesLeaf_) continue;
                
                float leftMean = leftSum / leftCount;
                float rightMean = rightSum / rightCount;
                
                float mse = 0.0f;
                for (int idx : indices) {
                    float pred = (X[idx * nFeatures_ + f] <= threshold) ? leftMean : rightMean;
                    mse += (y[idx] - pred) * (y[idx] - pred);
                }
                
                if (mse < bestMSE) {
                    bestMSE = mse;
                    bestFeature = f;
                    bestThreshold = threshold;
                }
            }
        }
        
        // Split
        std::vector<int> leftIndices, rightIndices;
        for (int idx : indices) {
            if (X[idx * nFeatures_ + bestFeature] <= bestThreshold) {
                leftIndices.push_back(idx);
            } else {
                rightIndices.push_back(idx);
            }
        }
        
        if (leftIndices.empty() || rightIndices.empty()) {
            node->isLeaf = true;
            node->prediction = mean;
            return node;
        }
        
        node->featureIdx = bestFeature;
        node->threshold = bestThreshold;
        node->left = buildTree(X, y, leftIndices, depth + 1);
        node->right = buildTree(X, y, rightIndices, depth + 1);
        
        return node;
    }
    
    std::unique_ptr<Node> root_;
    int maxDepth_;
    int minSamplesSplit_;
    int minSamplesLeaf_;
    int nFeatures_ = 0;
};

} // namespace ml
} // namespace neurova

#endif // NEUROVA_ML_REGRESSION_HPP
