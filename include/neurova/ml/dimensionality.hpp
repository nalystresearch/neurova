// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file dimensionality.hpp
 * @brief Dimensionality reduction algorithms
 */

#ifndef NEUROVA_ML_DIMENSIONALITY_HPP
#define NEUROVA_ML_DIMENSIONALITY_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace neurova {
namespace ml {

/**
 * @brief Principal Component Analysis (PCA)
 */
class PCA {
public:
    PCA(int nComponents = 2) : nComponents_(nComponents) {}
    
    void fit(const float* X, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        if (nComponents_ <= 0 || nComponents_ > nFeatures) {
            nComponents_ = nFeatures;
        }
        
        // Compute mean
        mean_.resize(nFeatures, 0.0f);
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nFeatures; ++j) {
                mean_[j] += X[i * nFeatures + j];
            }
        }
        for (int j = 0; j < nFeatures; ++j) {
            mean_[j] /= nSamples;
        }
        
        // Center data
        std::vector<float> centered(nSamples * nFeatures);
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nFeatures; ++j) {
                centered[i * nFeatures + j] = X[i * nFeatures + j] - mean_[j];
            }
        }
        
        // Compute covariance matrix
        std::vector<float> cov(nFeatures * nFeatures, 0.0f);
        for (int i = 0; i < nFeatures; ++i) {
            for (int j = 0; j < nFeatures; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < nSamples; ++k) {
                    sum += centered[k * nFeatures + i] * centered[k * nFeatures + j];
                }
                cov[i * nFeatures + j] = sum / (nSamples - 1);
            }
        }
        
        // Power iteration to find principal components
        components_.resize(nComponents_ * nFeatures);
        explainedVariance_.resize(nComponents_);
        
        std::vector<float> deflatedCov = cov;
        
        for (int c = 0; c < nComponents_; ++c) {
            // Random initialization
            std::vector<float> v(nFeatures);
            for (int i = 0; i < nFeatures; ++i) {
                v[i] = static_cast<float>((i + c + 1) % 10) / 10.0f;
            }
            normalize(v);
            
            // Power iteration
            for (int iter = 0; iter < 100; ++iter) {
                std::vector<float> vNew(nFeatures, 0.0f);
                for (int i = 0; i < nFeatures; ++i) {
                    for (int j = 0; j < nFeatures; ++j) {
                        vNew[i] += deflatedCov[i * nFeatures + j] * v[j];
                    }
                }
                normalize(vNew);
                
                // Check convergence
                float diff = 0.0f;
                for (int i = 0; i < nFeatures; ++i) {
                    diff += (vNew[i] - v[i]) * (vNew[i] - v[i]);
                }
                v = vNew;
                if (diff < 1e-10f) break;
            }
            
            // Store component
            for (int i = 0; i < nFeatures; ++i) {
                components_[c * nFeatures + i] = v[i];
            }
            
            // Compute eigenvalue (explained variance)
            float eigenvalue = 0.0f;
            for (int i = 0; i < nFeatures; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < nFeatures; ++j) {
                    sum += deflatedCov[i * nFeatures + j] * v[j];
                }
                eigenvalue += v[i] * sum;
            }
            explainedVariance_[c] = eigenvalue;
            
            // Deflate covariance matrix
            for (int i = 0; i < nFeatures; ++i) {
                for (int j = 0; j < nFeatures; ++j) {
                    deflatedCov[i * nFeatures + j] -= eigenvalue * v[i] * v[j];
                }
            }
        }
        
        // Compute explained variance ratio
        float totalVariance = 0.0f;
        for (int i = 0; i < nFeatures; ++i) {
            totalVariance += cov[i * nFeatures + i];
        }
        
        explainedVarianceRatio_.resize(nComponents_);
        for (int c = 0; c < nComponents_; ++c) {
            explainedVarianceRatio_[c] = explainedVariance_[c] / totalVariance;
        }
    }
    
    void transform(const float* X, float* Xt, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            for (int c = 0; c < nComponents_; ++c) {
                float sum = 0.0f;
                for (int j = 0; j < nFeatures_; ++j) {
                    sum += (X[i * nFeatures_ + j] - mean_[j]) * components_[c * nFeatures_ + j];
                }
                Xt[i * nComponents_ + c] = sum;
            }
        }
    }
    
    void fitTransform(const float* X, float* Xt, int nSamples, int nFeatures) {
        fit(X, nSamples, nFeatures);
        transform(X, Xt, nSamples);
    }
    
    void inverseTransform(const float* Xt, float* X, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nFeatures_; ++j) {
                float sum = mean_[j];
                for (int c = 0; c < nComponents_; ++c) {
                    sum += Xt[i * nComponents_ + c] * components_[c * nFeatures_ + j];
                }
                X[i * nFeatures_ + j] = sum;
            }
        }
    }
    
    const std::vector<float>& components() const { return components_; }
    const std::vector<float>& explainedVariance() const { return explainedVariance_; }
    const std::vector<float>& explainedVarianceRatio() const { return explainedVarianceRatio_; }
    const std::vector<float>& mean() const { return mean_; }
    int nComponents() const { return nComponents_; }
    
private:
    void normalize(std::vector<float>& v) const {
        float norm = 0.0f;
        for (float x : v) norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (float& x : v) x /= norm;
        }
    }
    
    int nComponents_;
    int nFeatures_ = 0;
    std::vector<float> mean_;
    std::vector<float> components_;
    std::vector<float> explainedVariance_;
    std::vector<float> explainedVarianceRatio_;
};

/**
 * @brief Linear Discriminant Analysis (LDA)
 */
class LDA {
public:
    LDA(int nComponents = 2) : nComponents_(nComponents) {}
    
    void fit(const float* X, const int* y, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        
        // Find classes
        std::vector<int> classes(y, y + nSamples);
        std::sort(classes.begin(), classes.end());
        classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
        int nClasses = static_cast<int>(classes.size());
        
        if (nComponents_ <= 0 || nComponents_ >= nClasses) {
            nComponents_ = nClasses - 1;
        }
        
        // Compute overall mean
        mean_.resize(nFeatures, 0.0f);
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nFeatures; ++j) {
                mean_[j] += X[i * nFeatures + j];
            }
        }
        for (int j = 0; j < nFeatures; ++j) {
            mean_[j] /= nSamples;
        }
        
        // Compute class means and counts
        std::vector<std::vector<float>> classMeans(nClasses, std::vector<float>(nFeatures, 0.0f));
        std::vector<int> classCounts(nClasses, 0);
        
        std::map<int, int> classToIdx;
        for (size_t i = 0; i < classes.size(); ++i) {
            classToIdx[classes[i]] = static_cast<int>(i);
        }
        
        for (int i = 0; i < nSamples; ++i) {
            int c = classToIdx[y[i]];
            classCounts[c]++;
            for (int j = 0; j < nFeatures; ++j) {
                classMeans[c][j] += X[i * nFeatures + j];
            }
        }
        
        for (int c = 0; c < nClasses; ++c) {
            if (classCounts[c] > 0) {
                for (int j = 0; j < nFeatures; ++j) {
                    classMeans[c][j] /= classCounts[c];
                }
            }
        }
        
        // Compute within-class scatter matrix (Sw)
        std::vector<float> Sw(nFeatures * nFeatures, 0.0f);
        for (int i = 0; i < nSamples; ++i) {
            int c = classToIdx[y[i]];
            for (int j = 0; j < nFeatures; ++j) {
                float dj = X[i * nFeatures + j] - classMeans[c][j];
                for (int k = 0; k < nFeatures; ++k) {
                    float dk = X[i * nFeatures + k] - classMeans[c][k];
                    Sw[j * nFeatures + k] += dj * dk;
                }
            }
        }
        
        // Compute between-class scatter matrix (Sb)
        std::vector<float> Sb(nFeatures * nFeatures, 0.0f);
        for (int c = 0; c < nClasses; ++c) {
            for (int j = 0; j < nFeatures; ++j) {
                float dj = classMeans[c][j] - mean_[j];
                for (int k = 0; k < nFeatures; ++k) {
                    float dk = classMeans[c][k] - mean_[k];
                    Sb[j * nFeatures + k] += classCounts[c] * dj * dk;
                }
            }
        }
        
        // Simplified: use PCA on between-class scatter weighted by inverse within-class
        // For full LDA, would need to solve generalized eigenvalue problem
        // Here we approximate using Sb directly
        
        components_.resize(nComponents_ * nFeatures);
        
        std::vector<float> workMatrix = Sb;
        
        for (int comp = 0; comp < nComponents_; ++comp) {
            std::vector<float> v(nFeatures);
            for (int i = 0; i < nFeatures; ++i) {
                v[i] = static_cast<float>((i + comp + 1) % 10) / 10.0f;
            }
            normalize(v);
            
            for (int iter = 0; iter < 100; ++iter) {
                std::vector<float> vNew(nFeatures, 0.0f);
                for (int i = 0; i < nFeatures; ++i) {
                    for (int j = 0; j < nFeatures; ++j) {
                        vNew[i] += workMatrix[i * nFeatures + j] * v[j];
                    }
                }
                normalize(vNew);
                
                float diff = 0.0f;
                for (int i = 0; i < nFeatures; ++i) {
                    diff += (vNew[i] - v[i]) * (vNew[i] - v[i]);
                }
                v = vNew;
                if (diff < 1e-10f) break;
            }
            
            for (int i = 0; i < nFeatures; ++i) {
                components_[comp * nFeatures + i] = v[i];
            }
            
            // Deflate
            float eigenvalue = 0.0f;
            for (int i = 0; i < nFeatures; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < nFeatures; ++j) {
                    sum += workMatrix[i * nFeatures + j] * v[j];
                }
                eigenvalue += v[i] * sum;
            }
            
            for (int i = 0; i < nFeatures; ++i) {
                for (int j = 0; j < nFeatures; ++j) {
                    workMatrix[i * nFeatures + j] -= eigenvalue * v[i] * v[j];
                }
            }
        }
    }
    
    void transform(const float* X, float* Xt, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            for (int c = 0; c < nComponents_; ++c) {
                float sum = 0.0f;
                for (int j = 0; j < nFeatures_; ++j) {
                    sum += (X[i * nFeatures_ + j] - mean_[j]) * components_[c * nFeatures_ + j];
                }
                Xt[i * nComponents_ + c] = sum;
            }
        }
    }
    
private:
    void normalize(std::vector<float>& v) const {
        float norm = 0.0f;
        for (float x : v) norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (float& x : v) x /= norm;
        }
    }
    
    int nComponents_;
    int nFeatures_ = 0;
    std::vector<float> mean_;
    std::vector<float> components_;
};

/**
 * @brief t-SNE (t-distributed Stochastic Neighbor Embedding)
 * Simplified implementation for visualization
 */
class TSNE {
public:
    TSNE(int nComponents = 2, float perplexity = 30.0f, int maxIter = 1000, float learningRate = 200.0f)
        : nComponents_(nComponents), perplexity_(perplexity), maxIter_(maxIter), learningRate_(learningRate) {}
    
    void fitTransform(const float* X, float* Y, int nSamples, int nFeatures) {
        nSamples_ = nSamples;
        
        // Compute pairwise distances
        std::vector<float> distances(nSamples * nSamples);
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nSamples; ++j) {
                float dist = 0.0f;
                for (int k = 0; k < nFeatures; ++k) {
                    float diff = X[i * nFeatures + k] - X[j * nFeatures + k];
                    dist += diff * diff;
                }
                distances[i * nSamples + j] = dist;
            }
        }
        
        // Compute perplexity-based probabilities
        std::vector<float> P(nSamples * nSamples, 0.0f);
        
        for (int i = 0; i < nSamples; ++i) {
            // Binary search for sigma
            float sigmaMin = 1e-10f, sigmaMax = 1e10f;
            float sigma = 1.0f;
            
            for (int iter = 0; iter < 50; ++iter) {
                float sumP = 0.0f;
                for (int j = 0; j < nSamples; ++j) {
                    if (i != j) {
                        P[i * nSamples + j] = std::exp(-distances[i * nSamples + j] / (2 * sigma * sigma));
                        sumP += P[i * nSamples + j];
                    }
                }
                
                if (sumP > 0) {
                    for (int j = 0; j < nSamples; ++j) {
                        P[i * nSamples + j] /= sumP;
                    }
                }
                
                // Compute perplexity
                float entropy = 0.0f;
                for (int j = 0; j < nSamples; ++j) {
                    if (P[i * nSamples + j] > 1e-10f) {
                        entropy -= P[i * nSamples + j] * std::log2(P[i * nSamples + j]);
                    }
                }
                float currentPerp = std::pow(2.0f, entropy);
                
                if (std::abs(currentPerp - perplexity_) < 1e-5f) break;
                
                if (currentPerp > perplexity_) {
                    sigmaMax = sigma;
                    sigma = (sigma + sigmaMin) / 2;
                } else {
                    sigmaMin = sigma;
                    sigma = (sigma + sigmaMax) / 2;
                }
            }
        }
        
        // Symmetrize
        for (int i = 0; i < nSamples; ++i) {
            for (int j = i + 1; j < nSamples; ++j) {
                float pij = (P[i * nSamples + j] + P[j * nSamples + i]) / (2 * nSamples);
                P[i * nSamples + j] = P[j * nSamples + i] = pij;
            }
        }
        
        // Initialize Y randomly
        unsigned int seed = 42;
        for (int i = 0; i < nSamples * nComponents_; ++i) {
            seed = seed * 1103515245 + 12345;
            Y[i] = (static_cast<float>(seed % 10000) / 10000.0f - 0.5f) * 0.01f;
        }
        
        // Gradient descent
        std::vector<float> gains(nSamples * nComponents_, 1.0f);
        std::vector<float> velocities(nSamples * nComponents_, 0.0f);
        
        for (int iter = 0; iter < maxIter_; ++iter) {
            // Compute Q (t-distribution)
            std::vector<float> Q(nSamples * nSamples, 0.0f);
            float sumQ = 0.0f;
            
            for (int i = 0; i < nSamples; ++i) {
                for (int j = i + 1; j < nSamples; ++j) {
                    float dist = 0.0f;
                    for (int k = 0; k < nComponents_; ++k) {
                        float diff = Y[i * nComponents_ + k] - Y[j * nComponents_ + k];
                        dist += diff * diff;
                    }
                    float qij = 1.0f / (1.0f + dist);
                    Q[i * nSamples + j] = Q[j * nSamples + i] = qij;
                    sumQ += 2 * qij;
                }
            }
            
            if (sumQ > 0) {
                for (int i = 0; i < nSamples * nSamples; ++i) {
                    Q[i] /= sumQ;
                }
            }
            
            // Compute gradients
            std::vector<float> grad(nSamples * nComponents_, 0.0f);
            
            for (int i = 0; i < nSamples; ++i) {
                for (int j = 0; j < nSamples; ++j) {
                    if (i == j) continue;
                    
                    float dist = 0.0f;
                    for (int k = 0; k < nComponents_; ++k) {
                        float diff = Y[i * nComponents_ + k] - Y[j * nComponents_ + k];
                        dist += diff * diff;
                    }
                    
                    float mult = 4 * (P[i * nSamples + j] - Q[i * nSamples + j]) / (1 + dist);
                    
                    for (int k = 0; k < nComponents_; ++k) {
                        grad[i * nComponents_ + k] += mult * (Y[i * nComponents_ + k] - Y[j * nComponents_ + k]);
                    }
                }
            }
            
            // Update with momentum
            float momentum = (iter < 250) ? 0.5f : 0.8f;
            
            for (int i = 0; i < nSamples * nComponents_; ++i) {
                // Adaptive gains
                if ((grad[i] > 0) != (velocities[i] > 0)) {
                    gains[i] = std::min(gains[i] + 0.2f, 10.0f);
                } else {
                    gains[i] = std::max(gains[i] * 0.8f, 0.01f);
                }
                
                velocities[i] = momentum * velocities[i] - learningRate_ * gains[i] * grad[i];
                Y[i] += velocities[i];
            }
            
            // Center
            std::vector<float> mean(nComponents_, 0.0f);
            for (int i = 0; i < nSamples; ++i) {
                for (int k = 0; k < nComponents_; ++k) {
                    mean[k] += Y[i * nComponents_ + k];
                }
            }
            for (int k = 0; k < nComponents_; ++k) {
                mean[k] /= nSamples;
            }
            for (int i = 0; i < nSamples; ++i) {
                for (int k = 0; k < nComponents_; ++k) {
                    Y[i * nComponents_ + k] -= mean[k];
                }
            }
        }
    }
    
private:
    int nComponents_;
    float perplexity_;
    int maxIter_;
    float learningRate_;
    int nSamples_ = 0;
};

/**
 * @brief Truncated SVD (for sparse data or LSA)
 */
class TruncatedSVD {
public:
    TruncatedSVD(int nComponents = 2) : nComponents_(nComponents) {}
    
    void fit(const float* X, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        
        // Power iteration for finding top singular vectors
        components_.resize(nComponents_ * nFeatures);
        singularValues_.resize(nComponents_);
        
        // Compute X^T * X
        std::vector<float> XTX(nFeatures * nFeatures, 0.0f);
        for (int i = 0; i < nFeatures; ++i) {
            for (int j = 0; j < nFeatures; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < nSamples; ++k) {
                    sum += X[k * nFeatures + i] * X[k * nFeatures + j];
                }
                XTX[i * nFeatures + j] = sum;
            }
        }
        
        std::vector<float> work = XTX;
        
        for (int c = 0; c < nComponents_; ++c) {
            std::vector<float> v(nFeatures);
            for (int i = 0; i < nFeatures; ++i) {
                v[i] = static_cast<float>((i + c + 1) % 10) / 10.0f;
            }
            normalize(v);
            
            for (int iter = 0; iter < 100; ++iter) {
                std::vector<float> vNew(nFeatures, 0.0f);
                for (int i = 0; i < nFeatures; ++i) {
                    for (int j = 0; j < nFeatures; ++j) {
                        vNew[i] += work[i * nFeatures + j] * v[j];
                    }
                }
                normalize(vNew);
                
                float diff = 0.0f;
                for (int i = 0; i < nFeatures; ++i) {
                    diff += (vNew[i] - v[i]) * (vNew[i] - v[i]);
                }
                v = vNew;
                if (diff < 1e-10f) break;
            }
            
            for (int i = 0; i < nFeatures; ++i) {
                components_[c * nFeatures + i] = v[i];
            }
            
            // Compute eigenvalue (singular value squared)
            float eigenvalue = 0.0f;
            for (int i = 0; i < nFeatures; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < nFeatures; ++j) {
                    sum += work[i * nFeatures + j] * v[j];
                }
                eigenvalue += v[i] * sum;
            }
            singularValues_[c] = std::sqrt(eigenvalue);
            
            // Deflate
            for (int i = 0; i < nFeatures; ++i) {
                for (int j = 0; j < nFeatures; ++j) {
                    work[i * nFeatures + j] -= eigenvalue * v[i] * v[j];
                }
            }
        }
    }
    
    void transform(const float* X, float* Xt, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            for (int c = 0; c < nComponents_; ++c) {
                float sum = 0.0f;
                for (int j = 0; j < nFeatures_; ++j) {
                    sum += X[i * nFeatures_ + j] * components_[c * nFeatures_ + j];
                }
                Xt[i * nComponents_ + c] = sum;
            }
        }
    }
    
    void fitTransform(const float* X, float* Xt, int nSamples, int nFeatures) {
        fit(X, nSamples, nFeatures);
        transform(X, Xt, nSamples);
    }
    
    const std::vector<float>& components() const { return components_; }
    const std::vector<float>& singularValues() const { return singularValues_; }
    
private:
    void normalize(std::vector<float>& v) const {
        float norm = 0.0f;
        for (float x : v) norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (float& x : v) x /= norm;
        }
    }
    
    int nComponents_;
    int nFeatures_ = 0;
    std::vector<float> components_;
    std::vector<float> singularValues_;
};

} // namespace ml
} // namespace neurova

#endif // NEUROVA_ML_DIMENSIONALITY_HPP
