// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file clustering.hpp
 * @brief Clustering algorithms
 */

#ifndef NEUROVA_ML_CLUSTERING_HPP
#define NEUROVA_ML_CLUSTERING_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <map>
#include <queue>

namespace neurova {
namespace ml {

/**
 * @brief K-Means clustering
 */
class KMeans {
public:
    KMeans(int nClusters = 8, int maxIter = 300, float tol = 1e-4f, unsigned int seed = 42)
        : nClusters_(nClusters), maxIter_(maxIter), tol_(tol), seed_(seed) {}
    
    void fit(const float* X, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        centers_.resize(nClusters_ * nFeatures);
        labels_.resize(nSamples);
        
        // Initialize centers using k-means++
        initCenters(X, nSamples, nFeatures);
        
        // Iterate
        for (int iter = 0; iter < maxIter_; ++iter) {
            // Assign labels
            for (int i = 0; i < nSamples; ++i) {
                labels_[i] = nearestCenter(X + i * nFeatures);
            }
            
            // Update centers
            std::vector<float> newCenters(nClusters_ * nFeatures, 0.0f);
            std::vector<int> counts(nClusters_, 0);
            
            for (int i = 0; i < nSamples; ++i) {
                int c = labels_[i];
                counts[c]++;
                for (int j = 0; j < nFeatures; ++j) {
                    newCenters[c * nFeatures + j] += X[i * nFeatures + j];
                }
            }
            
            for (int c = 0; c < nClusters_; ++c) {
                if (counts[c] > 0) {
                    for (int j = 0; j < nFeatures; ++j) {
                        newCenters[c * nFeatures + j] /= counts[c];
                    }
                }
            }
            
            // Check convergence
            float maxShift = 0.0f;
            for (int c = 0; c < nClusters_; ++c) {
                float shift = 0.0f;
                for (int j = 0; j < nFeatures; ++j) {
                    float diff = newCenters[c * nFeatures + j] - centers_[c * nFeatures + j];
                    shift += diff * diff;
                }
                maxShift = std::max(maxShift, std::sqrt(shift));
            }
            
            centers_ = newCenters;
            
            if (maxShift < tol_) break;
        }
        
        // Compute inertia
        inertia_ = 0.0f;
        for (int i = 0; i < nSamples; ++i) {
            float dist = distance(X + i * nFeatures, centers_.data() + labels_[i] * nFeatures, nFeatures);
            inertia_ += dist * dist;
        }
    }
    
    int predict(const float* sample) const {
        return nearestCenter(sample);
    }
    
    void predict(const float* X, int* predictions, int nSamples) const {
        for (int i = 0; i < nSamples; ++i) {
            predictions[i] = predict(X + i * nFeatures_);
        }
    }
    
    void fitPredict(const float* X, int* labels, int nSamples, int nFeatures) {
        fit(X, nSamples, nFeatures);
        std::copy(labels_.begin(), labels_.end(), labels);
    }
    
    const std::vector<float>& clusterCenters() const { return centers_; }
    const std::vector<int>& labels() const { return labels_; }
    float inertia() const { return inertia_; }
    
private:
    void initCenters(const float* X, int nSamples, int nFeatures) {
        unsigned int state = seed_;
        
        // First center: random
        state = state * 1103515245 + 12345;
        int firstIdx = (state / 65536) % nSamples;
        for (int j = 0; j < nFeatures; ++j) {
            centers_[j] = X[firstIdx * nFeatures + j];
        }
        
        // Remaining centers: k-means++
        std::vector<float> minDistSq(nSamples);
        
        for (int c = 1; c < nClusters_; ++c) {
            // Compute distances to nearest center
            float totalDist = 0.0f;
            for (int i = 0; i < nSamples; ++i) {
                float minDist = std::numeric_limits<float>::max();
                for (int k = 0; k < c; ++k) {
                    float dist = distance(X + i * nFeatures, centers_.data() + k * nFeatures, nFeatures);
                    minDist = std::min(minDist, dist);
                }
                minDistSq[i] = minDist * minDist;
                totalDist += minDistSq[i];
            }
            
            // Choose new center proportional to squared distance
            state = state * 1103515245 + 12345;
            float r = (state % 10000) / 10000.0f * totalDist;
            float cumSum = 0.0f;
            int newCenter = 0;
            for (int i = 0; i < nSamples; ++i) {
                cumSum += minDistSq[i];
                if (cumSum >= r) {
                    newCenter = i;
                    break;
                }
            }
            
            for (int j = 0; j < nFeatures; ++j) {
                centers_[c * nFeatures + j] = X[newCenter * nFeatures + j];
            }
        }
    }
    
    int nearestCenter(const float* sample) const {
        int best = 0;
        float bestDist = std::numeric_limits<float>::max();
        
        for (int c = 0; c < nClusters_; ++c) {
            float dist = distance(sample, centers_.data() + c * nFeatures_, nFeatures_);
            if (dist < bestDist) {
                bestDist = dist;
                best = c;
            }
        }
        
        return best;
    }
    
    static float distance(const float* a, const float* b, int n) {
        float dist = 0.0f;
        for (int i = 0; i < n; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }
    
    int nClusters_;
    int maxIter_;
    float tol_;
    unsigned int seed_;
    std::vector<float> centers_;
    std::vector<int> labels_;
    float inertia_ = 0.0f;
    int nFeatures_ = 0;
};

/**
 * @brief Mini-Batch K-Means
 */
class MiniBatchKMeans {
public:
    MiniBatchKMeans(int nClusters = 8, int batchSize = 100, int maxIter = 100, unsigned int seed = 42)
        : nClusters_(nClusters), batchSize_(batchSize), maxIter_(maxIter), seed_(seed) {}
    
    void fit(const float* X, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        centers_.resize(nClusters_ * nFeatures);
        
        // Initialize centers randomly
        unsigned int state = seed_;
        for (int c = 0; c < nClusters_; ++c) {
            state = state * 1103515245 + 12345;
            int idx = (state / 65536) % nSamples;
            for (int j = 0; j < nFeatures; ++j) {
                centers_[c * nFeatures + j] = X[idx * nFeatures + j];
            }
        }
        
        std::vector<int> counts(nClusters_, 0);
        
        for (int iter = 0; iter < maxIter_; ++iter) {
            // Sample mini-batch
            std::vector<int> batchIndices(batchSize_);
            for (int b = 0; b < batchSize_; ++b) {
                state = state * 1103515245 + 12345;
                batchIndices[b] = (state / 65536) % nSamples;
            }
            
            // Assign labels
            std::vector<int> batchLabels(batchSize_);
            for (int b = 0; b < batchSize_; ++b) {
                const float* sample = X + batchIndices[b] * nFeatures;
                float minDist = std::numeric_limits<float>::max();
                
                for (int c = 0; c < nClusters_; ++c) {
                    float dist = 0.0f;
                    for (int j = 0; j < nFeatures; ++j) {
                        float diff = sample[j] - centers_[c * nFeatures + j];
                        dist += diff * diff;
                    }
                    if (dist < minDist) {
                        minDist = dist;
                        batchLabels[b] = c;
                    }
                }
            }
            
            // Update centers
            for (int b = 0; b < batchSize_; ++b) {
                int c = batchLabels[b];
                counts[c]++;
                float lr = 1.0f / counts[c];
                
                const float* sample = X + batchIndices[b] * nFeatures;
                for (int j = 0; j < nFeatures; ++j) {
                    centers_[c * nFeatures + j] += lr * (sample[j] - centers_[c * nFeatures + j]);
                }
            }
        }
    }
    
    int predict(const float* sample) const {
        int best = 0;
        float bestDist = std::numeric_limits<float>::max();
        
        for (int c = 0; c < nClusters_; ++c) {
            float dist = 0.0f;
            for (int j = 0; j < nFeatures_; ++j) {
                float diff = sample[j] - centers_[c * nFeatures_ + j];
                dist += diff * diff;
            }
            if (dist < bestDist) {
                bestDist = dist;
                best = c;
            }
        }
        
        return best;
    }
    
private:
    int nClusters_;
    int batchSize_;
    int maxIter_;
    unsigned int seed_;
    std::vector<float> centers_;
    int nFeatures_ = 0;
};

/**
 * @brief DBSCAN clustering
 */
class DBSCAN {
public:
    DBSCAN(float eps = 0.5f, int minSamples = 5)
        : eps_(eps), minSamples_(minSamples) {}
    
    void fit(const float* X, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        labels_.assign(nSamples, -1);  // -1 = noise
        
        int clusterId = 0;
        
        for (int i = 0; i < nSamples; ++i) {
            if (labels_[i] != -1) continue;  // Already visited
            
            // Find neighbors
            std::vector<int> neighbors = regionQuery(X, nSamples, nFeatures, i);
            
            if (static_cast<int>(neighbors.size()) < minSamples_) {
                labels_[i] = -1;  // Noise
            } else {
                // Start new cluster
                expandCluster(X, nSamples, nFeatures, i, neighbors, clusterId);
                clusterId++;
            }
        }
        
        nClusters_ = clusterId;
    }
    
    const std::vector<int>& labels() const { return labels_; }
    int nClusters() const { return nClusters_; }
    
private:
    std::vector<int> regionQuery(const float* X, int nSamples, int nFeatures, int idx) const {
        std::vector<int> neighbors;
        const float* point = X + idx * nFeatures;
        
        for (int i = 0; i < nSamples; ++i) {
            float dist = 0.0f;
            for (int j = 0; j < nFeatures; ++j) {
                float diff = point[j] - X[i * nFeatures + j];
                dist += diff * diff;
            }
            if (std::sqrt(dist) <= eps_) {
                neighbors.push_back(i);
            }
        }
        
        return neighbors;
    }
    
    void expandCluster(const float* X, int nSamples, int nFeatures,
                      int pointIdx, std::vector<int>& neighbors, int clusterId) {
        labels_[pointIdx] = clusterId;
        
        std::queue<int> queue;
        for (int n : neighbors) queue.push(n);
        
        while (!queue.empty()) {
            int current = queue.front();
            queue.pop();
            
            if (labels_[current] == -1) {
                labels_[current] = clusterId;  // Was noise, now border point
            }
            
            if (labels_[current] != -1 && labels_[current] != clusterId) continue;
            
            labels_[current] = clusterId;
            
            auto newNeighbors = regionQuery(X, nSamples, nFeatures, current);
            if (static_cast<int>(newNeighbors.size()) >= minSamples_) {
                for (int n : newNeighbors) {
                    if (labels_[n] == -1) {
                        queue.push(n);
                    }
                }
            }
        }
    }
    
    float eps_;
    int minSamples_;
    std::vector<int> labels_;
    int nClusters_ = 0;
    int nFeatures_ = 0;
};

/**
 * @brief Agglomerative clustering
 */
class AgglomerativeClustering {
public:
    enum Linkage { SINGLE, COMPLETE, AVERAGE, WARD };
    
    AgglomerativeClustering(int nClusters = 2, Linkage linkage = WARD)
        : nClusters_(nClusters), linkage_(linkage) {}
    
    void fit(const float* X, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        labels_.resize(nSamples);
        
        // Initialize each point as its own cluster
        std::vector<std::vector<int>> clusters(nSamples);
        for (int i = 0; i < nSamples; ++i) {
            clusters[i].push_back(i);
        }
        
        // Compute initial distance matrix
        std::vector<std::vector<float>> distMatrix(nSamples, std::vector<float>(nSamples, 0.0f));
        for (int i = 0; i < nSamples; ++i) {
            for (int j = i + 1; j < nSamples; ++j) {
                float dist = 0.0f;
                for (int k = 0; k < nFeatures; ++k) {
                    float diff = X[i * nFeatures + k] - X[j * nFeatures + k];
                    dist += diff * diff;
                }
                distMatrix[i][j] = distMatrix[j][i] = std::sqrt(dist);
            }
        }
        
        // Merge clusters until we have nClusters_
        std::vector<bool> active(nSamples, true);
        int numActive = nSamples;
        
        while (numActive > nClusters_) {
            // Find closest pair
            float minDist = std::numeric_limits<float>::max();
            int minI = -1, minJ = -1;
            
            for (int i = 0; i < nSamples; ++i) {
                if (!active[i]) continue;
                for (int j = i + 1; j < nSamples; ++j) {
                    if (!active[j]) continue;
                    
                    float dist = computeLinkageDistance(X, nFeatures, clusters[i], clusters[j], distMatrix);
                    if (dist < minDist) {
                        minDist = dist;
                        minI = i;
                        minJ = j;
                    }
                }
            }
            
            // Merge clusters
            for (int idx : clusters[minJ]) {
                clusters[minI].push_back(idx);
            }
            clusters[minJ].clear();
            active[minJ] = false;
            numActive--;
        }
        
        // Assign labels
        int label = 0;
        for (int i = 0; i < nSamples; ++i) {
            if (active[i] && !clusters[i].empty()) {
                for (int idx : clusters[i]) {
                    labels_[idx] = label;
                }
                label++;
            }
        }
    }
    
    const std::vector<int>& labels() const { return labels_; }
    
private:
    float computeLinkageDistance(const float* X, int nFeatures,
                                 const std::vector<int>& c1, const std::vector<int>& c2,
                                 const std::vector<std::vector<float>>& distMatrix) const {
        if (c1.empty() || c2.empty()) return std::numeric_limits<float>::max();
        
        float result = 0.0f;
        
        switch (linkage_) {
            case SINGLE: {
                result = std::numeric_limits<float>::max();
                for (int i : c1) {
                    for (int j : c2) {
                        result = std::min(result, distMatrix[i][j]);
                    }
                }
                break;
            }
            case COMPLETE: {
                result = 0.0f;
                for (int i : c1) {
                    for (int j : c2) {
                        result = std::max(result, distMatrix[i][j]);
                    }
                }
                break;
            }
            case AVERAGE: {
                result = 0.0f;
                for (int i : c1) {
                    for (int j : c2) {
                        result += distMatrix[i][j];
                    }
                }
                result /= (c1.size() * c2.size());
                break;
            }
            case WARD: {
                // Compute centroids
                std::vector<float> centroid1(nFeatures, 0.0f);
                std::vector<float> centroid2(nFeatures, 0.0f);
                
                for (int i : c1) {
                    for (int k = 0; k < nFeatures; ++k) {
                        centroid1[k] += X[i * nFeatures + k];
                    }
                }
                for (int i : c2) {
                    for (int k = 0; k < nFeatures; ++k) {
                        centroid2[k] += X[i * nFeatures + k];
                    }
                }
                for (int k = 0; k < nFeatures; ++k) {
                    centroid1[k] /= c1.size();
                    centroid2[k] /= c2.size();
                }
                
                // Ward distance
                float n1 = static_cast<float>(c1.size());
                float n2 = static_cast<float>(c2.size());
                
                float dist = 0.0f;
                for (int k = 0; k < nFeatures; ++k) {
                    float diff = centroid1[k] - centroid2[k];
                    dist += diff * diff;
                }
                
                result = std::sqrt((n1 * n2) / (n1 + n2) * dist);
                break;
            }
        }
        
        return result;
    }
    
    int nClusters_;
    Linkage linkage_;
    std::vector<int> labels_;
    int nFeatures_ = 0;
};

/**
 * @brief Mean Shift clustering
 */
class MeanShift {
public:
    MeanShift(float bandwidth = 1.0f, int maxIter = 300)
        : bandwidth_(bandwidth), maxIter_(maxIter) {}
    
    void fit(const float* X, int nSamples, int nFeatures) {
        nFeatures_ = nFeatures;
        
        // Copy data points as initial seeds
        std::vector<std::vector<float>> seeds(nSamples);
        for (int i = 0; i < nSamples; ++i) {
            seeds[i].assign(X + i * nFeatures, X + (i + 1) * nFeatures);
        }
        
        // Shift each seed
        for (int i = 0; i < nSamples; ++i) {
            for (int iter = 0; iter < maxIter_; ++iter) {
                std::vector<float> newSeed(nFeatures, 0.0f);
                float totalWeight = 0.0f;
                
                for (int j = 0; j < nSamples; ++j) {
                    float dist = 0.0f;
                    for (int k = 0; k < nFeatures; ++k) {
                        float diff = seeds[i][k] - X[j * nFeatures + k];
                        dist += diff * diff;
                    }
                    dist = std::sqrt(dist);
                    
                    if (dist < bandwidth_) {
                        float weight = std::exp(-0.5f * (dist / bandwidth_) * (dist / bandwidth_));
                        for (int k = 0; k < nFeatures; ++k) {
                            newSeed[k] += weight * X[j * nFeatures + k];
                        }
                        totalWeight += weight;
                    }
                }
                
                if (totalWeight > 0) {
                    for (int k = 0; k < nFeatures; ++k) {
                        newSeed[k] /= totalWeight;
                    }
                }
                
                // Check convergence
                float shift = 0.0f;
                for (int k = 0; k < nFeatures; ++k) {
                    float diff = newSeed[k] - seeds[i][k];
                    shift += diff * diff;
                }
                
                seeds[i] = newSeed;
                
                if (std::sqrt(shift) < 1e-4f) break;
            }
        }
        
        // Cluster converged points
        std::vector<std::vector<float>> uniqueCenters;
        labels_.resize(nSamples);
        
        for (int i = 0; i < nSamples; ++i) {
            // Find closest existing center
            int bestCenter = -1;
            float minDist = bandwidth_ / 2.0f;
            
            for (size_t c = 0; c < uniqueCenters.size(); ++c) {
                float dist = 0.0f;
                for (int k = 0; k < nFeatures; ++k) {
                    float diff = seeds[i][k] - uniqueCenters[c][k];
                    dist += diff * diff;
                }
                dist = std::sqrt(dist);
                if (dist < minDist) {
                    minDist = dist;
                    bestCenter = static_cast<int>(c);
                }
            }
            
            if (bestCenter < 0) {
                bestCenter = static_cast<int>(uniqueCenters.size());
                uniqueCenters.push_back(seeds[i]);
            }
            
            labels_[i] = bestCenter;
        }
        
        centers_ = uniqueCenters;
    }
    
    const std::vector<int>& labels() const { return labels_; }
    const std::vector<std::vector<float>>& clusterCenters() const { return centers_; }
    
private:
    float bandwidth_;
    int maxIter_;
    std::vector<int> labels_;
    std::vector<std::vector<float>> centers_;
    int nFeatures_ = 0;
};

/**
 * @brief Spectral clustering (simplified)
 */
class SpectralClustering {
public:
    SpectralClustering(int nClusters = 2, float gamma = 1.0f)
        : nClusters_(nClusters), gamma_(gamma) {}
    
    void fit(const float* X, int nSamples, int nFeatures) {
        // Build affinity matrix using RBF kernel
        std::vector<float> affinity(nSamples * nSamples);
        
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nSamples; ++j) {
                if (i == j) {
                    affinity[i * nSamples + j] = 0.0f;
                } else {
                    float dist = 0.0f;
                    for (int k = 0; k < nFeatures; ++k) {
                        float diff = X[i * nFeatures + k] - X[j * nFeatures + k];
                        dist += diff * diff;
                    }
                    affinity[i * nSamples + j] = std::exp(-gamma_ * dist);
                }
            }
        }
        
        // Compute degree matrix and normalized Laplacian
        std::vector<float> degrees(nSamples, 0.0f);
        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nSamples; ++j) {
                degrees[i] += affinity[i * nSamples + j];
            }
        }
        
        // Simplified: just use affinity weighted features for K-means
        std::vector<float> embedding(nSamples * nFeatures);
        for (int i = 0; i < nSamples; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < nSamples; ++j) {
                sum += affinity[i * nSamples + j];
            }
            
            for (int k = 0; k < nFeatures; ++k) {
                float val = 0.0f;
                for (int j = 0; j < nSamples; ++j) {
                    val += affinity[i * nSamples + j] * X[j * nFeatures + k];
                }
                embedding[i * nFeatures + k] = (sum > 0) ? val / sum : X[i * nFeatures + k];
            }
        }
        
        // K-means on embedding
        KMeans kmeans(nClusters_);
        kmeans.fit(embedding.data(), nSamples, nFeatures);
        labels_ = kmeans.labels();
    }
    
    const std::vector<int>& labels() const { return labels_; }
    
private:
    int nClusters_;
    float gamma_;
    std::vector<int> labels_;
};

} // namespace ml
} // namespace neurova

#endif // NEUROVA_ML_CLUSTERING_HPP
