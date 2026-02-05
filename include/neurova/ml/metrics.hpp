// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file metrics.hpp
 * @brief Machine learning evaluation metrics
 */

#ifndef NEUROVA_ML_METRICS_HPP
#define NEUROVA_ML_METRICS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>

namespace neurova {
namespace ml {
namespace metrics {

// ============================================================================
// Classification Metrics
// ============================================================================

/**
 * @brief Compute accuracy score
 */
inline float accuracyScore(const int* yTrue, const int* yPred, int nSamples) {
    int correct = 0;
    for (int i = 0; i < nSamples; ++i) {
        if (yTrue[i] == yPred[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct) / nSamples;
}

/**
 * @brief Compute confusion matrix
 */
inline std::vector<std::vector<int>> confusionMatrix(const int* yTrue, const int* yPred, int nSamples) {
    // Find unique classes
    std::set<int> classSet;
    for (int i = 0; i < nSamples; ++i) {
        classSet.insert(yTrue[i]);
        classSet.insert(yPred[i]);
    }
    
    std::vector<int> classes(classSet.begin(), classSet.end());
    int nClasses = static_cast<int>(classes.size());
    
    std::map<int, int> classToIdx;
    for (int i = 0; i < nClasses; ++i) {
        classToIdx[classes[i]] = i;
    }
    
    std::vector<std::vector<int>> cm(nClasses, std::vector<int>(nClasses, 0));
    
    for (int i = 0; i < nSamples; ++i) {
        int trueIdx = classToIdx[yTrue[i]];
        int predIdx = classToIdx[yPred[i]];
        cm[trueIdx][predIdx]++;
    }
    
    return cm;
}

/**
 * @brief Classification report metrics for a single class
 */
struct ClassMetrics {
    float precision;
    float recall;
    float f1Score;
    int support;
};

/**
 * @brief Compute precision score (binary or macro-averaged)
 */
inline float precisionScore(const int* yTrue, const int* yPred, int nSamples, 
                           int posLabel = 1, bool macroAvg = false) {
    if (macroAvg) {
        // Macro-averaged precision
        std::set<int> classes;
        for (int i = 0; i < nSamples; ++i) {
            classes.insert(yTrue[i]);
        }
        
        float sumPrecision = 0.0f;
        int nClasses = 0;
        
        for (int cls : classes) {
            int tp = 0, fp = 0;
            for (int i = 0; i < nSamples; ++i) {
                if (yPred[i] == cls) {
                    if (yTrue[i] == cls) tp++;
                    else fp++;
                }
            }
            if (tp + fp > 0) {
                sumPrecision += static_cast<float>(tp) / (tp + fp);
            }
            nClasses++;
        }
        
        return sumPrecision / nClasses;
    } else {
        // Binary precision for positive label
        int tp = 0, fp = 0;
        for (int i = 0; i < nSamples; ++i) {
            if (yPred[i] == posLabel) {
                if (yTrue[i] == posLabel) tp++;
                else fp++;
            }
        }
        return (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
    }
}

/**
 * @brief Compute recall score (binary or macro-averaged)
 */
inline float recallScore(const int* yTrue, const int* yPred, int nSamples,
                        int posLabel = 1, bool macroAvg = false) {
    if (macroAvg) {
        std::set<int> classes;
        for (int i = 0; i < nSamples; ++i) {
            classes.insert(yTrue[i]);
        }
        
        float sumRecall = 0.0f;
        int nClasses = 0;
        
        for (int cls : classes) {
            int tp = 0, fn = 0;
            for (int i = 0; i < nSamples; ++i) {
                if (yTrue[i] == cls) {
                    if (yPred[i] == cls) tp++;
                    else fn++;
                }
            }
            if (tp + fn > 0) {
                sumRecall += static_cast<float>(tp) / (tp + fn);
            }
            nClasses++;
        }
        
        return sumRecall / nClasses;
    } else {
        int tp = 0, fn = 0;
        for (int i = 0; i < nSamples; ++i) {
            if (yTrue[i] == posLabel) {
                if (yPred[i] == posLabel) tp++;
                else fn++;
            }
        }
        return (tp + fn > 0) ? static_cast<float>(tp) / (tp + fn) : 0.0f;
    }
}

/**
 * @brief Compute F1 score (binary or macro-averaged)
 */
inline float f1Score(const int* yTrue, const int* yPred, int nSamples,
                    int posLabel = 1, bool macroAvg = false) {
    float p = precisionScore(yTrue, yPred, nSamples, posLabel, macroAvg);
    float r = recallScore(yTrue, yPred, nSamples, posLabel, macroAvg);
    return (p + r > 0) ? 2 * p * r / (p + r) : 0.0f;
}

/**
 * @brief Compute F-beta score
 */
inline float fbetaScore(const int* yTrue, const int* yPred, int nSamples,
                       float beta, int posLabel = 1) {
    float p = precisionScore(yTrue, yPred, nSamples, posLabel);
    float r = recallScore(yTrue, yPred, nSamples, posLabel);
    float beta2 = beta * beta;
    return (p + r > 0) ? (1 + beta2) * p * r / (beta2 * p + r) : 0.0f;
}

/**
 * @brief Generate classification report
 */
inline std::map<int, ClassMetrics> classificationReport(const int* yTrue, const int* yPred, int nSamples) {
    std::set<int> classes;
    for (int i = 0; i < nSamples; ++i) {
        classes.insert(yTrue[i]);
    }
    
    std::map<int, ClassMetrics> report;
    
    for (int cls : classes) {
        int tp = 0, fp = 0, fn = 0;
        
        for (int i = 0; i < nSamples; ++i) {
            if (yTrue[i] == cls && yPred[i] == cls) tp++;
            else if (yTrue[i] != cls && yPred[i] == cls) fp++;
            else if (yTrue[i] == cls && yPred[i] != cls) fn++;
        }
        
        ClassMetrics metrics;
        metrics.precision = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
        metrics.recall = (tp + fn > 0) ? static_cast<float>(tp) / (tp + fn) : 0.0f;
        metrics.f1Score = (metrics.precision + metrics.recall > 0) 
            ? 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall) 
            : 0.0f;
        metrics.support = tp + fn;
        
        report[cls] = metrics;
    }
    
    return report;
}

/**
 * @brief Compute log loss (cross-entropy loss)
 */
inline float logLoss(const int* yTrue, const float* yProba, int nSamples, int nClasses) {
    float loss = 0.0f;
    const float eps = 1e-15f;
    
    for (int i = 0; i < nSamples; ++i) {
        int trueClass = yTrue[i];
        float p = yProba[i * nClasses + trueClass];
        p = std::max(eps, std::min(1.0f - eps, p));
        loss -= std::log(p);
    }
    
    return loss / nSamples;
}

/**
 * @brief Compute ROC curve points
 */
inline void rocCurve(const int* yTrue, const float* yScore, int nSamples,
                    std::vector<float>& fpr, std::vector<float>& tpr, 
                    std::vector<float>& thresholds) {
    // Sort by scores
    std::vector<std::pair<float, int>> scoredLabels(nSamples);
    for (int i = 0; i < nSamples; ++i) {
        scoredLabels[i] = {yScore[i], yTrue[i]};
    }
    std::sort(scoredLabels.begin(), scoredLabels.end(), std::greater<>());
    
    // Count positives and negatives
    int P = 0, N = 0;
    for (int i = 0; i < nSamples; ++i) {
        if (yTrue[i] == 1) P++;
        else N++;
    }
    
    fpr.clear();
    tpr.clear();
    thresholds.clear();
    
    fpr.push_back(0.0f);
    tpr.push_back(0.0f);
    thresholds.push_back(1.1f);
    
    int tp = 0, fp = 0;
    float prevScore = 1e10f;
    
    for (int i = 0; i < nSamples; ++i) {
        if (scoredLabels[i].first != prevScore) {
            fpr.push_back(static_cast<float>(fp) / N);
            tpr.push_back(static_cast<float>(tp) / P);
            thresholds.push_back(scoredLabels[i].first);
            prevScore = scoredLabels[i].first;
        }
        
        if (scoredLabels[i].second == 1) tp++;
        else fp++;
    }
    
    fpr.push_back(1.0f);
    tpr.push_back(1.0f);
    thresholds.push_back(0.0f);
}

/**
 * @brief Compute AUC-ROC score
 */
inline float rocAucScore(const int* yTrue, const float* yScore, int nSamples) {
    std::vector<float> fpr, tpr, thresholds;
    rocCurve(yTrue, yScore, nSamples, fpr, tpr, thresholds);
    
    // Trapezoidal integration
    float auc = 0.0f;
    for (size_t i = 1; i < fpr.size(); ++i) {
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2;
    }
    
    return auc;
}

// ============================================================================
// Regression Metrics
// ============================================================================

/**
 * @brief Mean Squared Error
 */
inline float meanSquaredError(const float* yTrue, const float* yPred, int nSamples) {
    float mse = 0.0f;
    for (int i = 0; i < nSamples; ++i) {
        float diff = yTrue[i] - yPred[i];
        mse += diff * diff;
    }
    return mse / nSamples;
}

/**
 * @brief Root Mean Squared Error
 */
inline float rootMeanSquaredError(const float* yTrue, const float* yPred, int nSamples) {
    return std::sqrt(meanSquaredError(yTrue, yPred, nSamples));
}

/**
 * @brief Mean Absolute Error
 */
inline float meanAbsoluteError(const float* yTrue, const float* yPred, int nSamples) {
    float mae = 0.0f;
    for (int i = 0; i < nSamples; ++i) {
        mae += std::abs(yTrue[i] - yPred[i]);
    }
    return mae / nSamples;
}

/**
 * @brief R² score (coefficient of determination)
 */
inline float r2Score(const float* yTrue, const float* yPred, int nSamples) {
    float meanTrue = 0.0f;
    for (int i = 0; i < nSamples; ++i) {
        meanTrue += yTrue[i];
    }
    meanTrue /= nSamples;
    
    float ssRes = 0.0f, ssTot = 0.0f;
    for (int i = 0; i < nSamples; ++i) {
        float diff = yTrue[i] - yPred[i];
        ssRes += diff * diff;
        float diffMean = yTrue[i] - meanTrue;
        ssTot += diffMean * diffMean;
    }
    
    return (ssTot > 0) ? 1.0f - ssRes / ssTot : 0.0f;
}

/**
 * @brief Adjusted R² score
 */
inline float adjustedR2Score(const float* yTrue, const float* yPred, int nSamples, int nFeatures) {
    float r2 = r2Score(yTrue, yPred, nSamples);
    return 1.0f - (1.0f - r2) * (nSamples - 1) / (nSamples - nFeatures - 1);
}

/**
 * @brief Mean Absolute Percentage Error
 */
inline float meanAbsolutePercentageError(const float* yTrue, const float* yPred, int nSamples) {
    float mape = 0.0f;
    int count = 0;
    for (int i = 0; i < nSamples; ++i) {
        if (std::abs(yTrue[i]) > 1e-10f) {
            mape += std::abs((yTrue[i] - yPred[i]) / yTrue[i]);
            count++;
        }
    }
    return (count > 0) ? 100.0f * mape / count : 0.0f;
}

/**
 * @brief Max Error
 */
inline float maxError(const float* yTrue, const float* yPred, int nSamples) {
    float maxErr = 0.0f;
    for (int i = 0; i < nSamples; ++i) {
        maxErr = std::max(maxErr, std::abs(yTrue[i] - yPred[i]));
    }
    return maxErr;
}

/**
 * @brief Explained Variance Score
 */
inline float explainedVarianceScore(const float* yTrue, const float* yPred, int nSamples) {
    // Compute residuals
    std::vector<float> residuals(nSamples);
    for (int i = 0; i < nSamples; ++i) {
        residuals[i] = yTrue[i] - yPred[i];
    }
    
    // Variance of yTrue
    float meanTrue = 0.0f;
    for (int i = 0; i < nSamples; ++i) {
        meanTrue += yTrue[i];
    }
    meanTrue /= nSamples;
    
    float varTrue = 0.0f;
    for (int i = 0; i < nSamples; ++i) {
        float diff = yTrue[i] - meanTrue;
        varTrue += diff * diff;
    }
    varTrue /= nSamples;
    
    // Variance of residuals
    float meanRes = 0.0f;
    for (int i = 0; i < nSamples; ++i) {
        meanRes += residuals[i];
    }
    meanRes /= nSamples;
    
    float varRes = 0.0f;
    for (int i = 0; i < nSamples; ++i) {
        float diff = residuals[i] - meanRes;
        varRes += diff * diff;
    }
    varRes /= nSamples;
    
    return (varTrue > 0) ? 1.0f - varRes / varTrue : 0.0f;
}

// ============================================================================
// Clustering Metrics
// ============================================================================

/**
 * @brief Compute silhouette score for clustering
 */
inline float silhouetteScore(const float* X, const int* labels, int nSamples, int nFeatures) {
    if (nSamples < 2) return 0.0f;
    
    // Find unique labels
    std::set<int> labelSet(labels, labels + nSamples);
    if (labelSet.size() < 2) return 0.0f;
    
    std::vector<float> silhouettes(nSamples);
    
    for (int i = 0; i < nSamples; ++i) {
        int clusterI = labels[i];
        
        // Compute a(i) - mean intra-cluster distance
        float sumIntra = 0.0f;
        int countIntra = 0;
        
        // Compute b(i) - mean nearest-cluster distance
        std::map<int, float> clusterDists;
        std::map<int, int> clusterCounts;
        
        for (int j = 0; j < nSamples; ++j) {
            if (i == j) continue;
            
            float dist = 0.0f;
            for (int k = 0; k < nFeatures; ++k) {
                float diff = X[i * nFeatures + k] - X[j * nFeatures + k];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);
            
            if (labels[j] == clusterI) {
                sumIntra += dist;
                countIntra++;
            } else {
                clusterDists[labels[j]] += dist;
                clusterCounts[labels[j]]++;
            }
        }
        
        float a = (countIntra > 0) ? sumIntra / countIntra : 0.0f;
        
        float b = std::numeric_limits<float>::max();
        for (auto& kv : clusterDists) {
            if (clusterCounts[kv.first] > 0) {
                float meanDist = kv.second / clusterCounts[kv.first];
                b = std::min(b, meanDist);
            }
        }
        
        if (b == std::numeric_limits<float>::max()) b = 0.0f;
        
        float maxAB = std::max(a, b);
        silhouettes[i] = (maxAB > 0) ? (b - a) / maxAB : 0.0f;
    }
    
    float meanSilhouette = 0.0f;
    for (float s : silhouettes) {
        meanSilhouette += s;
    }
    return meanSilhouette / nSamples;
}

/**
 * @brief Compute Davies-Bouldin Index for clustering
 */
inline float daviesBouldinScore(const float* X, const int* labels, int nSamples, int nFeatures) {
    std::set<int> labelSet(labels, labels + nSamples);
    std::vector<int> uniqueLabels(labelSet.begin(), labelSet.end());
    int nClusters = static_cast<int>(uniqueLabels.size());
    
    if (nClusters < 2) return 0.0f;
    
    std::map<int, int> labelToIdx;
    for (int i = 0; i < nClusters; ++i) {
        labelToIdx[uniqueLabels[i]] = i;
    }
    
    // Compute cluster centroids
    std::vector<std::vector<float>> centroids(nClusters, std::vector<float>(nFeatures, 0.0f));
    std::vector<int> clusterCounts(nClusters, 0);
    
    for (int i = 0; i < nSamples; ++i) {
        int idx = labelToIdx[labels[i]];
        clusterCounts[idx]++;
        for (int k = 0; k < nFeatures; ++k) {
            centroids[idx][k] += X[i * nFeatures + k];
        }
    }
    
    for (int c = 0; c < nClusters; ++c) {
        if (clusterCounts[c] > 0) {
            for (int k = 0; k < nFeatures; ++k) {
                centroids[c][k] /= clusterCounts[c];
            }
        }
    }
    
    // Compute intra-cluster scatter (average distance to centroid)
    std::vector<float> scatter(nClusters, 0.0f);
    for (int i = 0; i < nSamples; ++i) {
        int idx = labelToIdx[labels[i]];
        float dist = 0.0f;
        for (int k = 0; k < nFeatures; ++k) {
            float diff = X[i * nFeatures + k] - centroids[idx][k];
            dist += diff * diff;
        }
        scatter[idx] += std::sqrt(dist);
    }
    
    for (int c = 0; c < nClusters; ++c) {
        if (clusterCounts[c] > 0) {
            scatter[c] /= clusterCounts[c];
        }
    }
    
    // Compute DBI
    float dbi = 0.0f;
    for (int i = 0; i < nClusters; ++i) {
        float maxRatio = 0.0f;
        for (int j = 0; j < nClusters; ++j) {
            if (i == j) continue;
            
            float centroidDist = 0.0f;
            for (int k = 0; k < nFeatures; ++k) {
                float diff = centroids[i][k] - centroids[j][k];
                centroidDist += diff * diff;
            }
            centroidDist = std::sqrt(centroidDist);
            
            if (centroidDist > 0) {
                float ratio = (scatter[i] + scatter[j]) / centroidDist;
                maxRatio = std::max(maxRatio, ratio);
            }
        }
        dbi += maxRatio;
    }
    
    return dbi / nClusters;
}

/**
 * @brief Compute Calinski-Harabasz Index (Variance Ratio Criterion)
 */
inline float calinskiHarabaszScore(const float* X, const int* labels, int nSamples, int nFeatures) {
    std::set<int> labelSet(labels, labels + nSamples);
    std::vector<int> uniqueLabels(labelSet.begin(), labelSet.end());
    int nClusters = static_cast<int>(uniqueLabels.size());
    
    if (nClusters < 2 || nSamples <= nClusters) return 0.0f;
    
    std::map<int, int> labelToIdx;
    for (int i = 0; i < nClusters; ++i) {
        labelToIdx[uniqueLabels[i]] = i;
    }
    
    // Compute overall centroid
    std::vector<float> globalCentroid(nFeatures, 0.0f);
    for (int i = 0; i < nSamples; ++i) {
        for (int k = 0; k < nFeatures; ++k) {
            globalCentroid[k] += X[i * nFeatures + k];
        }
    }
    for (int k = 0; k < nFeatures; ++k) {
        globalCentroid[k] /= nSamples;
    }
    
    // Compute cluster centroids
    std::vector<std::vector<float>> centroids(nClusters, std::vector<float>(nFeatures, 0.0f));
    std::vector<int> clusterCounts(nClusters, 0);
    
    for (int i = 0; i < nSamples; ++i) {
        int idx = labelToIdx[labels[i]];
        clusterCounts[idx]++;
        for (int k = 0; k < nFeatures; ++k) {
            centroids[idx][k] += X[i * nFeatures + k];
        }
    }
    
    for (int c = 0; c < nClusters; ++c) {
        if (clusterCounts[c] > 0) {
            for (int k = 0; k < nFeatures; ++k) {
                centroids[c][k] /= clusterCounts[c];
            }
        }
    }
    
    // Between-cluster dispersion
    float bgss = 0.0f;
    for (int c = 0; c < nClusters; ++c) {
        float dist = 0.0f;
        for (int k = 0; k < nFeatures; ++k) {
            float diff = centroids[c][k] - globalCentroid[k];
            dist += diff * diff;
        }
        bgss += clusterCounts[c] * dist;
    }
    
    // Within-cluster dispersion
    float wgss = 0.0f;
    for (int i = 0; i < nSamples; ++i) {
        int idx = labelToIdx[labels[i]];
        float dist = 0.0f;
        for (int k = 0; k < nFeatures; ++k) {
            float diff = X[i * nFeatures + k] - centroids[idx][k];
            dist += diff * diff;
        }
        wgss += dist;
    }
    
    if (wgss < 1e-10f) return 0.0f;
    
    return (bgss / (nClusters - 1)) / (wgss / (nSamples - nClusters));
}

/**
 * @brief Compute inertia (sum of squared distances to nearest centroid)
 */
inline float inertia(const float* X, const int* labels, const float* centroids,
                    int nSamples, int nFeatures, int nClusters) {
    float totalInertia = 0.0f;
    
    for (int i = 0; i < nSamples; ++i) {
        int cluster = labels[i];
        float dist = 0.0f;
        for (int k = 0; k < nFeatures; ++k) {
            float diff = X[i * nFeatures + k] - centroids[cluster * nFeatures + k];
            dist += diff * diff;
        }
        totalInertia += dist;
    }
    
    return totalInertia;
}

// ============================================================================
// Distance Functions
// ============================================================================

/**
 * @brief Euclidean distance
 */
inline float euclideanDistance(const float* a, const float* b, int n) {
    float dist = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

/**
 * @brief Manhattan distance
 */
inline float manhattanDistance(const float* a, const float* b, int n) {
    float dist = 0.0f;
    for (int i = 0; i < n; ++i) {
        dist += std::abs(a[i] - b[i]);
    }
    return dist;
}

/**
 * @brief Cosine distance
 */
inline float cosineDistance(const float* a, const float* b, int n) {
    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for (int i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    float denom = std::sqrt(normA) * std::sqrt(normB);
    return (denom > 0) ? 1.0f - dot / denom : 1.0f;
}

/**
 * @brief Pairwise distance matrix
 */
inline void pairwiseDistances(const float* X, float* D, int nSamples, int nFeatures,
                             const std::string& metric = "euclidean") {
    for (int i = 0; i < nSamples; ++i) {
        D[i * nSamples + i] = 0.0f;
        for (int j = i + 1; j < nSamples; ++j) {
            float dist;
            if (metric == "manhattan") {
                dist = manhattanDistance(&X[i * nFeatures], &X[j * nFeatures], nFeatures);
            } else if (metric == "cosine") {
                dist = cosineDistance(&X[i * nFeatures], &X[j * nFeatures], nFeatures);
            } else {
                dist = euclideanDistance(&X[i * nFeatures], &X[j * nFeatures], nFeatures);
            }
            D[i * nSamples + j] = D[j * nSamples + i] = dist;
        }
    }
}

} // namespace metrics
} // namespace ml
} // namespace neurova

#endif // NEUROVA_ML_METRICS_HPP
