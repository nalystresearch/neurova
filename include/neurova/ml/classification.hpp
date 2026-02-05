// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file classification.hpp
 * @brief Classification algorithms
 */

#ifndef NEUROVA_ML_CLASSIFICATION_HPP
#define NEUROVA_ML_CLASSIFICATION_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <limits>
#include <queue>

namespace neurova {
namespace ml {

/**
 * @brief K-Nearest Neighbors classifier
 */
class KNeighborsClassifier {
public:
    KNeighborsClassifier(int k = 5, const std::string& weights = "uniform")
        : k_(k), weights_(weights) {}
    
    void fit(const float* X, const int* y, int nSamples, int nFeatures) {
        nSamples_ = nSamples;
        nFeatures_ = nFeatures;
        
        X_.assign(X, X + nSamples * nFeatures);
        y_.assign(y, y + nSamples);
    }
    
    int predict(const float* sample) const {
        // Find k nearest neighbors
        std::vector<std::pair<float, int>> distances;
        
        for (int i = 0; i < nSamples_; ++i) {
            float dist = 0.0f;
            for (int j = 0; j < nFeatures_; ++j) {
                float diff = sample[j] - X_[i * nFeatures_ + j];
                dist += diff * diff;
            }
            distances.push_back({std::sqrt(dist), y_[i]});
        }
        
        std::partial_sort(distances.begin(), distances.begin() + k_, distances.end());
        
        // Vote
        std::map<int, float> votes;
        for (int i = 0; i < k_; ++i) {
            float weight = 1.0f;
            if (weights_ == "distance" && distances[i].first > 0) {
                weight = 1.0f / distances[i].first;
            }
            votes[distances[i].second] += weight;
        }
        
        int bestClass = -1;
        float maxVotes = -1.0f;
        for (const auto& [cls, v] : votes) {
            if (v > maxVotes) {
                maxVotes = v;
                bestClass = cls;
            }
        }
        
        return bestClass;
    }
    
    void predict(const float* X, int* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
    std::vector<float> predictProba(const float* sample) const {
        // Find unique classes
        std::vector<int> classes(y_.begin(), y_.end());
        std::sort(classes.begin(), classes.end());
        classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
        
        std::map<int, int> classToIdx;
        for (size_t i = 0; i < classes.size(); ++i) {
            classToIdx[classes[i]] = static_cast<int>(i);
        }
        
        // Find k nearest neighbors
        std::vector<std::pair<float, int>> distances;
        for (int i = 0; i < nSamples_; ++i) {
            float dist = 0.0f;
            for (int j = 0; j < nFeatures_; ++j) {
                float diff = sample[j] - X_[i * nFeatures_ + j];
                dist += diff * diff;
            }
            distances.push_back({std::sqrt(dist), y_[i]});
        }
        
        std::partial_sort(distances.begin(), distances.begin() + k_, distances.end());
        
        // Compute probabilities
        std::vector<float> proba(classes.size(), 0.0f);
        float totalWeight = 0.0f;
        
        for (int i = 0; i < k_; ++i) {
            float weight = 1.0f;
            if (weights_ == "distance" && distances[i].first > 0) {
                weight = 1.0f / distances[i].first;
            }
            proba[classToIdx[distances[i].second]] += weight;
            totalWeight += weight;
        }
        
        if (totalWeight > 0) {
            for (float& p : proba) p /= totalWeight;
        }
        
        return proba;
    }
    
private:
    int k_;
    std::string weights_;
    std::vector<float> X_;
    std::vector<int> y_;
    int nSamples_ = 0;
    int nFeatures_ = 0;
};

/**
 * @brief Logistic Regression classifier
 */
class LogisticRegression {
public:
    LogisticRegression(float learningRate = 0.01f, int maxIter = 100, float regularization = 0.0f)
        : learningRate_(learningRate), maxIter_(maxIter), regularization_(regularization) {}
    
    void fit(const float* X, const int* y, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        
        // Find unique classes
        std::vector<int> classes(y, y + nSamples);
        std::sort(classes.begin(), classes.end());
        classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
        
        if (classes.size() == 2) {
            // Binary classification
            fitBinary(X, y, nSamples, nFeatures, classes[0], classes[1]);
        } else {
            // Multiclass: one-vs-rest
            nClasses_ = static_cast<int>(classes.size());
            classes_ = classes;
            weights_.resize(nClasses_ * (nFeatures + 1));
            
            for (int c = 0; c < nClasses_; ++c) {
                std::vector<int> binaryY(nSamples);
                for (int i = 0; i < nSamples; ++i) {
                    binaryY[i] = (y[i] == classes_[c]) ? 1 : 0;
                }
                
                std::vector<float> classWeights = fitBinaryReturnWeights(
                    X, binaryY.data(), nSamples, nFeatures);
                
                for (int j = 0; j <= nFeatures; ++j) {
                    weights_[c * (nFeatures + 1) + j] = classWeights[j];
                }
            }
        }
    }
    
    int predict(const float* sample) const {
        auto proba = predictProba(sample);
        int maxIdx = static_cast<int>(std::max_element(proba.begin(), proba.end()) - proba.begin());
        return classes_.empty() ? maxIdx : classes_[maxIdx];
    }
    
    void predict(const float* X, int* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
    std::vector<float> predictProba(const float* sample) const {
        if (nClasses_ == 2 || classes_.empty()) {
            float z = weights_[0];  // Bias
            for (int j = 0; j < nFeatures_; ++j) {
                z += weights_[j + 1] * sample[j];
            }
            float p = sigmoid(z);
            return {1.0f - p, p};
        }
        
        // Multiclass
        std::vector<float> proba(nClasses_);
        float sum = 0.0f;
        
        for (int c = 0; c < nClasses_; ++c) {
            float z = weights_[c * (nFeatures_ + 1)];  // Bias
            for (int j = 0; j < nFeatures_; ++j) {
                z += weights_[c * (nFeatures_ + 1) + j + 1] * sample[j];
            }
            proba[c] = std::exp(z);
            sum += proba[c];
        }
        
        for (float& p : proba) p /= sum;
        return proba;
    }
    
private:
    float sigmoid(float x) const {
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    void fitBinary(const float* X, const int* y, int nSamples, int nFeatures,
                   int class0, int class1) {
        classes_ = {class0, class1};
        nClasses_ = 2;
        weights_.resize(nFeatures + 1, 0.0f);
        
        for (int iter = 0; iter < maxIter_; ++iter) {
            std::vector<float> grad(nFeatures + 1, 0.0f);
            
            for (int i = 0; i < nSamples; ++i) {
                float z = weights_[0];
                for (int j = 0; j < nFeatures; ++j) {
                    z += weights_[j + 1] * X[i * nFeatures + j];
                }
                float p = sigmoid(z);
                float target = (y[i] == class1) ? 1.0f : 0.0f;
                float error = p - target;
                
                grad[0] += error;
                for (int j = 0; j < nFeatures; ++j) {
                    grad[j + 1] += error * X[i * nFeatures + j];
                }
            }
            
            // Update weights
            for (int j = 0; j <= nFeatures; ++j) {
                weights_[j] -= learningRate_ * (grad[j] / nSamples + regularization_ * weights_[j]);
            }
        }
    }
    
    std::vector<float> fitBinaryReturnWeights(const float* X, const int* y, 
                                               int nSamples, int nFeatures) {
        std::vector<float> w(nFeatures + 1, 0.0f);
        
        for (int iter = 0; iter < maxIter_; ++iter) {
            std::vector<float> grad(nFeatures + 1, 0.0f);
            
            for (int i = 0; i < nSamples; ++i) {
                float z = w[0];
                for (int j = 0; j < nFeatures; ++j) {
                    z += w[j + 1] * X[i * nFeatures + j];
                }
                float p = sigmoid(z);
                float error = p - y[i];
                
                grad[0] += error;
                for (int j = 0; j < nFeatures; ++j) {
                    grad[j + 1] += error * X[i * nFeatures + j];
                }
            }
            
            for (int j = 0; j <= nFeatures; ++j) {
                w[j] -= learningRate_ * (grad[j] / nSamples + regularization_ * w[j]);
            }
        }
        
        return w;
    }
    
    float learningRate_;
    int maxIter_;
    float regularization_;
    std::vector<float> weights_;
    std::vector<int> classes_;
    int nFeatures_ = 0;
    int nClasses_ = 0;
};

/**
 * @brief Gaussian Naive Bayes classifier
 */
class GaussianNB {
public:
    void fit(const float* X, const int* y, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        
        // Find unique classes
        std::vector<int> classes(y, y + nSamples);
        std::sort(classes.begin(), classes.end());
        classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
        classes_ = classes;
        nClasses_ = static_cast<int>(classes.size());
        
        // Compute class priors, means, and variances
        classPriors_.resize(nClasses_);
        means_.resize(nClasses_ * nFeatures);
        variances_.resize(nClasses_ * nFeatures);
        
        std::map<int, int> classToIdx;
        for (size_t i = 0; i < classes.size(); ++i) {
            classToIdx[classes[i]] = static_cast<int>(i);
        }
        
        std::vector<int> classCounts(nClasses_, 0);
        std::fill(means_.begin(), means_.end(), 0.0f);
        
        // Compute means
        for (int i = 0; i < nSamples; ++i) {
            int c = classToIdx[y[i]];
            classCounts[c]++;
            for (int j = 0; j < nFeatures; ++j) {
                means_[c * nFeatures + j] += X[i * nFeatures + j];
            }
        }
        
        for (int c = 0; c < nClasses_; ++c) {
            classPriors_[c] = static_cast<float>(classCounts[c]) / nSamples;
            if (classCounts[c] > 0) {
                for (int j = 0; j < nFeatures; ++j) {
                    means_[c * nFeatures + j] /= classCounts[c];
                }
            }
        }
        
        // Compute variances
        std::fill(variances_.begin(), variances_.end(), 0.0f);
        for (int i = 0; i < nSamples; ++i) {
            int c = classToIdx[y[i]];
            for (int j = 0; j < nFeatures; ++j) {
                float diff = X[i * nFeatures + j] - means_[c * nFeatures + j];
                variances_[c * nFeatures + j] += diff * diff;
            }
        }
        
        for (int c = 0; c < nClasses_; ++c) {
            for (int j = 0; j < nFeatures; ++j) {
                if (classCounts[c] > 1) {
                    variances_[c * nFeatures + j] /= classCounts[c];
                }
                // Add small value to prevent division by zero
                variances_[c * nFeatures + j] += 1e-9f;
            }
        }
    }
    
    int predict(const float* sample) const {
        auto logProba = computeLogProba(sample);
        int maxIdx = static_cast<int>(std::max_element(logProba.begin(), logProba.end()) - logProba.begin());
        return classes_[maxIdx];
    }
    
    void predict(const float* X, int* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
    std::vector<float> predictProba(const float* sample) const {
        auto logProba = computeLogProba(sample);
        
        // Convert log probabilities to probabilities
        float maxLog = *std::max_element(logProba.begin(), logProba.end());
        std::vector<float> proba(nClasses_);
        float sum = 0.0f;
        
        for (int c = 0; c < nClasses_; ++c) {
            proba[c] = std::exp(logProba[c] - maxLog);
            sum += proba[c];
        }
        
        for (float& p : proba) p /= sum;
        return proba;
    }
    
private:
    std::vector<float> computeLogProba(const float* sample) const {
        std::vector<float> logProba(nClasses_);
        
        for (int c = 0; c < nClasses_; ++c) {
            logProba[c] = std::log(classPriors_[c]);
            
            for (int j = 0; j < nFeatures_; ++j) {
                float mean = means_[c * nFeatures_ + j];
                float var = variances_[c * nFeatures_ + j];
                float diff = sample[j] - mean;
                
                // Log of Gaussian PDF
                logProba[c] -= 0.5f * (std::log(2.0f * M_PI * var) + diff * diff / var);
            }
        }
        
        return logProba;
    }
    
    std::vector<int> classes_;
    std::vector<float> classPriors_;
    std::vector<float> means_;
    std::vector<float> variances_;
    int nFeatures_ = 0;
    int nClasses_ = 0;
};

/**
 * @brief Decision Tree classifier
 */
class DecisionTreeClassifier {
public:
    DecisionTreeClassifier(int maxDepth = 10, int minSamplesSplit = 2)
        : maxDepth_(maxDepth), minSamplesSplit_(minSamplesSplit) {}
    
    void fit(const float* X, const int* y, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        
        // Find unique classes
        std::vector<int> classes(y, y + nSamples);
        std::sort(classes.begin(), classes.end());
        classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
        classes_ = classes;
        
        // Build tree
        std::vector<int> indices(nSamples);
        std::iota(indices.begin(), indices.end(), 0);
        
        root_ = buildTree(X, y, indices, 0);
    }
    
    int predict(const float* sample) const {
        const Node* node = root_.get();
        while (node && !node->isLeaf) {
            if (sample[node->featureIdx] <= node->threshold) {
                node = node->left.get();
            } else {
                node = node->right.get();
            }
        }
        return node ? node->prediction : -1;
    }
    
    void predict(const float* X, int* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
private:
    struct Node {
        bool isLeaf = false;
        int featureIdx = -1;
        float threshold = 0.0f;
        int prediction = -1;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };
    
    std::unique_ptr<Node> buildTree(const float* X, const int* y,
                                    const std::vector<int>& indices, int depth) {
        auto node = std::make_unique<Node>();
        
        if (indices.empty()) {
            node->isLeaf = true;
            return node;
        }
        
        // Check if all same class or stopping criteria
        std::map<int, int> classCounts;
        for (int idx : indices) {
            classCounts[y[idx]]++;
        }
        
        if (classCounts.size() == 1 || depth >= maxDepth_ || 
            static_cast<int>(indices.size()) < minSamplesSplit_) {
            node->isLeaf = true;
            int maxCount = 0;
            for (const auto& [cls, count] : classCounts) {
                if (count > maxCount) {
                    maxCount = count;
                    node->prediction = cls;
                }
            }
            return node;
        }
        
        // Find best split
        float bestGini = std::numeric_limits<float>::max();
        int bestFeature = 0;
        float bestThreshold = 0.0f;
        
        for (int f = 0; f < nFeatures_; ++f) {
            // Get unique values for this feature
            std::vector<float> values;
            for (int idx : indices) {
                values.push_back(X[idx * nFeatures_ + f]);
            }
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
            
            // Try thresholds between consecutive values
            for (size_t i = 0; i < values.size() - 1; ++i) {
                float threshold = (values[i] + values[i + 1]) / 2.0f;
                
                std::map<int, int> leftCounts, rightCounts;
                int leftTotal = 0, rightTotal = 0;
                
                for (int idx : indices) {
                    if (X[idx * nFeatures_ + f] <= threshold) {
                        leftCounts[y[idx]]++;
                        leftTotal++;
                    } else {
                        rightCounts[y[idx]]++;
                        rightTotal++;
                    }
                }
                
                if (leftTotal == 0 || rightTotal == 0) continue;
                
                float leftGini = 1.0f, rightGini = 1.0f;
                for (const auto& [cls, count] : leftCounts) {
                    float p = static_cast<float>(count) / leftTotal;
                    leftGini -= p * p;
                }
                for (const auto& [cls, count] : rightCounts) {
                    float p = static_cast<float>(count) / rightTotal;
                    rightGini -= p * p;
                }
                
                float gini = (leftTotal * leftGini + rightTotal * rightGini) / indices.size();
                
                if (gini < bestGini) {
                    bestGini = gini;
                    bestFeature = f;
                    bestThreshold = threshold;
                }
            }
        }
        
        // Split
        node->featureIdx = bestFeature;
        node->threshold = bestThreshold;
        
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
            int maxCount = 0;
            for (const auto& [cls, count] : classCounts) {
                if (count > maxCount) {
                    maxCount = count;
                    node->prediction = cls;
                }
            }
            return node;
        }
        
        node->left = buildTree(X, y, leftIndices, depth + 1);
        node->right = buildTree(X, y, rightIndices, depth + 1);
        
        return node;
    }
    
    std::unique_ptr<Node> root_;
    std::vector<int> classes_;
    int maxDepth_;
    int minSamplesSplit_;
    int nFeatures_ = 0;
};

/**
 * @brief Random Forest classifier (simplified)
 */
class RandomForestClassifier {
public:
    RandomForestClassifier(int nEstimators = 10, int maxDepth = 10, unsigned int seed = 42)
        : nEstimators_(nEstimators), maxDepth_(maxDepth), seed_(seed) {}
    
    void fit(const float* X, const int* y, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        trees_.clear();
        
        // Find classes
        std::vector<int> classes(y, y + nSamples);
        std::sort(classes.begin(), classes.end());
        classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
        classes_ = classes;
        
        unsigned int state = seed_;
        
        for (int t = 0; t < nEstimators_; ++t) {
            // Bootstrap sample
            std::vector<int> indices(nSamples);
            for (int i = 0; i < nSamples; ++i) {
                state = state * 1103515245 + 12345;
                indices[i] = (state / 65536) % nSamples;
            }
            
            std::vector<float> Xsample(nSamples * nFeatures);
            std::vector<int> ysample(nSamples);
            for (int i = 0; i < nSamples; ++i) {
                for (int j = 0; j < nFeatures; ++j) {
                    Xsample[i * nFeatures + j] = X[indices[i] * nFeatures + j];
                }
                ysample[i] = y[indices[i]];
            }
            
            // Train tree
            auto tree = std::make_unique<DecisionTreeClassifier>(maxDepth_);
            tree->fit(Xsample.data(), ysample.data(), nSamples, nFeatures);
            trees_.push_back(std::move(tree));
        }
    }
    
    int predict(const float* sample) const {
        std::map<int, int> votes;
        for (const auto& tree : trees_) {
            votes[tree->predict(sample)]++;
        }
        
        int maxVotes = 0;
        int prediction = -1;
        for (const auto& [cls, v] : votes) {
            if (v > maxVotes) {
                maxVotes = v;
                prediction = cls;
            }
        }
        return prediction;
    }
    
    void predict(const float* X, int* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
private:
    int nEstimators_;
    int maxDepth_;
    unsigned int seed_;
    std::vector<std::unique_ptr<DecisionTreeClassifier>> trees_;
    std::vector<int> classes_;
    int nFeatures_ = 0;
};

} // namespace ml
} // namespace neurova

#endif // NEUROVA_ML_CLASSIFICATION_HPP
