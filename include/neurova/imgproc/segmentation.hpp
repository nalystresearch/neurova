// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file segmentation.hpp
 * @brief Image segmentation algorithms
 */

#ifndef NEUROVA_IMGPROC_SEGMENTATION_HPP
#define NEUROVA_IMGPROC_SEGMENTATION_HPP

#include "contours.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <limits>
#include <map>
#include <set>

namespace neurova {
namespace imgproc {

// GrabCut modes
constexpr int GC_INIT_WITH_RECT = 0;
constexpr int GC_INIT_WITH_MASK = 1;
constexpr int GC_EVAL = 2;
constexpr int GC_EVAL_FREEZE_MODEL = 3;

// GrabCut mask values
constexpr int GC_BGD = 0;
constexpr int GC_FGD = 1;
constexpr int GC_PR_BGD = 2;
constexpr int GC_PR_FGD = 3;

// Distance transform types
constexpr int DIST_L1 = 1;
constexpr int DIST_L2 = 2;
constexpr int DIST_C = 3;

// Connected components connectivity
constexpr int CC_4 = 4;
constexpr int CC_8 = 8;

/**
 * @brief Connected components labeling
 */
inline int connectedComponents(
    const uint8_t* binary, int width, int height,
    int* labels,
    int connectivity = CC_8
) {
    std::fill(labels, labels + width * height, 0);
    
    int numLabels = 0;
    std::vector<int> parent(width * height + 1);
    
    // Union-Find functions
    auto find = [&parent](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];  // Path compression
            x = parent[x];
        }
        return x;
    };
    
    auto unite = [&parent, &find](int x, int y) {
        int px = find(x);
        int py = find(y);
        if (px != py) {
            parent[px] = py;
        }
    };
    
    // First pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            if (binary[idx] == 0) continue;
            
            std::vector<int> neighbors;
            
            // Check neighbors (already processed)
            if (x > 0 && binary[idx - 1] > 0) {
                neighbors.push_back(labels[idx - 1]);
            }
            if (y > 0 && binary[idx - width] > 0) {
                neighbors.push_back(labels[idx - width]);
            }
            if (connectivity == CC_8) {
                if (x > 0 && y > 0 && binary[idx - width - 1] > 0) {
                    neighbors.push_back(labels[idx - width - 1]);
                }
                if (x < width - 1 && y > 0 && binary[idx - width + 1] > 0) {
                    neighbors.push_back(labels[idx - width + 1]);
                }
            }
            
            if (neighbors.empty()) {
                // New label
                ++numLabels;
                labels[idx] = numLabels;
                parent[numLabels] = numLabels;
            } else {
                // Use minimum label
                int minLabel = *std::min_element(neighbors.begin(), neighbors.end());
                labels[idx] = minLabel;
                
                // Union all neighbors
                for (int n : neighbors) {
                    unite(n, minLabel);
                }
            }
        }
    }
    
    // Second pass - resolve equivalences
    std::map<int, int> labelMap;
    int finalNumLabels = 0;
    
    for (int i = 0; i < width * height; ++i) {
        if (labels[i] > 0) {
            int root = find(labels[i]);
            if (labelMap.find(root) == labelMap.end()) {
                labelMap[root] = ++finalNumLabels;
            }
            labels[i] = labelMap[root];
        }
    }
    
    return finalNumLabels;
}

/**
 * @brief Connected components with stats
 */
inline int connectedComponentsWithStats(
    const uint8_t* binary, int width, int height,
    int* labels,
    std::vector<Rect>& boundingBoxes,
    std::vector<int>& areas,
    std::vector<Point2f>& centroids,
    int connectivity = CC_8
) {
    int numLabels = connectedComponents(binary, width, height, labels, connectivity);
    
    // Calculate stats
    boundingBoxes.resize(numLabels + 1);
    areas.resize(numLabels + 1, 0);
    centroids.resize(numLabels + 1);
    
    std::vector<int> minX(numLabels + 1, width);
    std::vector<int> maxX(numLabels + 1, 0);
    std::vector<int> minY(numLabels + 1, height);
    std::vector<int> maxY(numLabels + 1, 0);
    std::vector<double> sumX(numLabels + 1, 0);
    std::vector<double> sumY(numLabels + 1, 0);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int label = labels[y * width + x];
            if (label > 0) {
                areas[label]++;
                sumX[label] += x;
                sumY[label] += y;
                minX[label] = std::min(minX[label], x);
                maxX[label] = std::max(maxX[label], x);
                minY[label] = std::min(minY[label], y);
                maxY[label] = std::max(maxY[label], y);
            }
        }
    }
    
    for (int i = 1; i <= numLabels; ++i) {
        boundingBoxes[i] = Rect(minX[i], minY[i], maxX[i] - minX[i] + 1, maxY[i] - minY[i] + 1);
        if (areas[i] > 0) {
            centroids[i] = Point2f(
                static_cast<float>(sumX[i] / areas[i]),
                static_cast<float>(sumY[i] / areas[i])
            );
        }
    }
    
    return numLabels;
}

/**
 * @brief Distance transform
 */
inline void distanceTransform(
    const uint8_t* binary, int width, int height,
    float* dist,
    int distType = DIST_L2
) {
    const float INF = std::numeric_limits<float>::max();
    
    // Initialize
    for (int i = 0; i < width * height; ++i) {
        dist[i] = (binary[i] > 0) ? INF : 0.0f;
    }
    
    if (distType == DIST_L1 || distType == DIST_C) {
        // Chamfer distance (two-pass)
        
        // Forward pass
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                if (dist[idx] == 0) continue;
                
                float minDist = INF;
                if (x > 0) minDist = std::min(minDist, dist[idx - 1] + 1);
                if (y > 0) minDist = std::min(minDist, dist[idx - width] + 1);
                if (distType == DIST_C) {
                    if (x > 0 && y > 0) minDist = std::min(minDist, dist[idx - width - 1] + 1);
                    if (x < width - 1 && y > 0) minDist = std::min(minDist, dist[idx - width + 1] + 1);
                }
                dist[idx] = minDist;
            }
        }
        
        // Backward pass
        for (int y = height - 1; y >= 0; --y) {
            for (int x = width - 1; x >= 0; --x) {
                int idx = y * width + x;
                if (dist[idx] == 0) continue;
                
                float minDist = dist[idx];
                if (x < width - 1) minDist = std::min(minDist, dist[idx + 1] + 1);
                if (y < height - 1) minDist = std::min(minDist, dist[idx + width] + 1);
                if (distType == DIST_C) {
                    if (x < width - 1 && y < height - 1) minDist = std::min(minDist, dist[idx + width + 1] + 1);
                    if (x > 0 && y < height - 1) minDist = std::min(minDist, dist[idx + width - 1] + 1);
                }
                dist[idx] = minDist;
            }
        }
    } else {
        // Euclidean distance using squared distance transform
        // Meijster algorithm (separable)
        
        std::vector<float> tempDist(width * height);
        
        // Phase 1: Scan columns
        for (int x = 0; x < width; ++x) {
            // Forward scan
            tempDist[x] = (binary[x] > 0) ? INF : 0;
            for (int y = 1; y < height; ++y) {
                int idx = y * width + x;
                if (binary[idx] > 0) {
                    tempDist[idx] = tempDist[idx - width] + 1;
                } else {
                    tempDist[idx] = 0;
                }
            }
            
            // Backward scan
            for (int y = height - 2; y >= 0; --y) {
                int idx = y * width + x;
                if (tempDist[idx + width] + 1 < tempDist[idx]) {
                    tempDist[idx] = tempDist[idx + width] + 1;
                }
            }
        }
        
        // Phase 2: Scan rows
        auto f = [](int x, int i, float gi) -> float {
            return (x - i) * (x - i) + gi * gi;
        };
        
        auto sep = [](int i, int u, float gi, float gu) -> float {
            return (u * u - i * i + gu * gu - gi * gi) / (2.0f * (u - i));
        };
        
        for (int y = 0; y < height; ++y) {
            std::vector<int> s(width);
            std::vector<float> t(width);
            int q = 0;
            s[0] = 0;
            t[0] = static_cast<float>(-width);
            
            for (int u = 1; u < width; ++u) {
                float gu = tempDist[y * width + u];
                while (q >= 0 && f(static_cast<int>(t[q]), s[q], tempDist[y * width + s[q]]) > 
                       f(static_cast<int>(t[q]), u, gu)) {
                    --q;
                }
                
                if (q < 0) {
                    q = 0;
                    s[0] = u;
                } else {
                    float w = sep(s[q], u, tempDist[y * width + s[q]], gu);
                    if (w < width) {
                        ++q;
                        s[q] = u;
                        t[q] = w;
                    }
                }
            }
            
            for (int u = width - 1; u >= 0; --u) {
                dist[y * width + u] = std::sqrt(f(u, s[q], tempDist[y * width + s[q]]));
                if (u == static_cast<int>(t[q])) {
                    --q;
                }
            }
        }
    }
}

/**
 * @brief Watershed segmentation
 */
inline void watershed(
    const float* image, int width, int height, int channels,
    int* markers
) {
    // Priority queue: (negative gradient, x, y)
    using PQElement = std::tuple<float, int, int>;
    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> pq;
    
    const int WSHED = -1;  // Watershed line marker
    const int INIT = -2;   // Not yet processed
    
    // Compute gradient magnitude
    std::vector<float> gradient(width * height);
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float gx = 0, gy = 0;
            for (int c = 0; c < channels; ++c) {
                float dx = image[(y * width + x + 1) * channels + c] -
                          image[(y * width + x - 1) * channels + c];
                float dy = image[((y + 1) * width + x) * channels + c] -
                          image[((y - 1) * width + x) * channels + c];
                gx += dx * dx;
                gy += dy * dy;
            }
            gradient[y * width + x] = std::sqrt(gx + gy);
        }
    }
    
    // Find initial labeled pixels and add their neighbors to queue
    std::vector<int> labels(width * height, INIT);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            if (markers[idx] > 0) {
                labels[idx] = markers[idx];
                
                // Add unlabeled neighbors to queue
                const int dx[] = {-1, 0, 1, 0};
                const int dy[] = {0, -1, 0, 1};
                
                for (int d = 0; d < 4; ++d) {
                    int nx = x + dx[d];
                    int ny = y + dy[d];
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int nidx = ny * width + nx;
                        if (labels[nidx] == INIT && markers[nidx] <= 0) {
                            pq.push({gradient[nidx], nx, ny});
                            labels[nidx] = 0;  // In queue
                        }
                    }
                }
            }
        }
    }
    
    // Process queue
    const int dx[] = {-1, 0, 1, 0};
    const int dy[] = {0, -1, 0, 1};
    
    while (!pq.empty()) {
        auto [grad, x, y] = pq.top();
        pq.pop();
        
        int idx = y * width + x;
        
        // Find labeled neighbors
        std::set<int> neighborLabels;
        for (int d = 0; d < 4; ++d) {
            int nx = x + dx[d];
            int ny = y + dy[d];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = ny * width + nx;
                if (labels[nidx] > 0) {
                    neighborLabels.insert(labels[nidx]);
                }
            }
        }
        
        if (neighborLabels.size() == 1) {
            // Single neighbor label - propagate
            labels[idx] = *neighborLabels.begin();
        } else if (neighborLabels.size() > 1) {
            // Multiple labels - watershed line
            labels[idx] = WSHED;
        }
        
        // Add unlabeled neighbors to queue
        for (int d = 0; d < 4; ++d) {
            int nx = x + dx[d];
            int ny = y + dy[d];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = ny * width + nx;
                if (labels[nidx] == INIT) {
                    pq.push({gradient[nidx], nx, ny});
                    labels[nidx] = 0;
                }
            }
        }
    }
    
    // Copy labels to markers
    for (int i = 0; i < width * height; ++i) {
        markers[i] = labels[i];
    }
}

/**
 * @brief Simple GrabCut segmentation (simplified implementation)
 */
inline void grabCut(
    const float* image, int width, int height, int channels,
    uint8_t* mask,
    Rect rect,
    int mode = GC_INIT_WITH_RECT,
    int iterCount = 5
) {
    // Initialize mask from rect
    if (mode == GC_INIT_WITH_RECT) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                if (x >= rect.x && x < rect.x + rect.width &&
                    y >= rect.y && y < rect.y + rect.height) {
                    mask[idx] = GC_PR_FGD;
                } else {
                    mask[idx] = GC_BGD;
                }
            }
        }
    }
    
    // Simple iterative region growing/shrinking based on color similarity
    for (int iter = 0; iter < iterCount; ++iter) {
        // Compute mean colors for foreground and background
        float fgMean[3] = {0, 0, 0};
        float bgMean[3] = {0, 0, 0};
        int fgCount = 0, bgCount = 0;
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                int mval = mask[idx];
                
                if (mval == GC_FGD || mval == GC_PR_FGD) {
                    for (int c = 0; c < std::min(channels, 3); ++c) {
                        fgMean[c] += image[idx * channels + c];
                    }
                    fgCount++;
                } else {
                    for (int c = 0; c < std::min(channels, 3); ++c) {
                        bgMean[c] += image[idx * channels + c];
                    }
                    bgCount++;
                }
            }
        }
        
        if (fgCount > 0) {
            for (int c = 0; c < 3; ++c) fgMean[c] /= fgCount;
        }
        if (bgCount > 0) {
            for (int c = 0; c < 3; ++c) bgMean[c] /= bgCount;
        }
        
        // Update probable pixels based on color distance
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                int mval = mask[idx];
                
                // Skip definite pixels
                if (mval == GC_FGD || mval == GC_BGD) continue;
                
                // Compute distance to fg/bg means
                float fgDist = 0, bgDist = 0;
                for (int c = 0; c < std::min(channels, 3); ++c) {
                    float v = image[idx * channels + c];
                    fgDist += (v - fgMean[c]) * (v - fgMean[c]);
                    bgDist += (v - bgMean[c]) * (v - bgMean[c]);
                }
                
                mask[idx] = (fgDist < bgDist) ? GC_PR_FGD : GC_PR_BGD;
            }
        }
    }
}

/**
 * @brief Flood fill operation
 */
inline int floodFill(
    float* image, int width, int height, int channels,
    int seedX, int seedY,
    const float* newVal,
    float loDiff = 0.0f,
    float upDiff = 0.0f,
    Rect* rect = nullptr
) {
    if (seedX < 0 || seedX >= width || seedY < 0 || seedY >= height) {
        return 0;
    }
    
    // Get seed color
    int seedIdx = seedY * width + seedX;
    std::vector<float> seedColor(channels);
    for (int c = 0; c < channels; ++c) {
        seedColor[c] = image[seedIdx * channels + c];
    }
    
    // BFS flood fill
    std::vector<bool> visited(width * height, false);
    std::queue<std::pair<int, int>> queue;
    queue.push({seedX, seedY});
    visited[seedIdx] = true;
    
    int filled = 0;
    int minX = seedX, maxX = seedX, minY = seedY, maxY = seedY;
    
    while (!queue.empty()) {
        auto [x, y] = queue.front();
        queue.pop();
        
        int idx = y * width + x;
        
        // Check if pixel matches criteria
        bool matches = true;
        for (int c = 0; c < channels && matches; ++c) {
            float diff = image[idx * channels + c] - seedColor[c];
            if (diff < -loDiff || diff > upDiff) {
                matches = false;
            }
        }
        
        if (!matches) continue;
        
        // Fill pixel
        for (int c = 0; c < channels; ++c) {
            image[idx * channels + c] = newVal[c];
        }
        filled++;
        
        minX = std::min(minX, x);
        maxX = std::max(maxX, x);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
        
        // Add neighbors
        const int dx[] = {-1, 1, 0, 0};
        const int dy[] = {0, 0, -1, 1};
        
        for (int d = 0; d < 4; ++d) {
            int nx = x + dx[d];
            int ny = y + dy[d];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = ny * width + nx;
                if (!visited[nidx]) {
                    visited[nidx] = true;
                    queue.push({nx, ny});
                }
            }
        }
    }
    
    if (rect) {
        *rect = Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
    }
    
    return filled;
}

/**
 * @brief Mean shift segmentation (simplified)
 */
inline void pyrMeanShiftFiltering(
    const float* src, float* dst, int width, int height, int channels,
    double sp,
    double sr,
    int maxLevel = 1
) {
    // For simplicity, single level mean shift
    std::vector<float> result(src, src + width * height * channels);
    
    int maxIter = 5;
    double epsilon = 1.0;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            
            double cx = x, cy = y;
            std::vector<double> cc(channels);
            for (int c = 0; c < channels; ++c) {
                cc[c] = src[idx * channels + c];
            }
            
            for (int iter = 0; iter < maxIter; ++iter) {
                double sumX = 0, sumY = 0;
                std::vector<double> sumC(channels, 0);
                double count = 0;
                
                // Search in spatial window
                int minSX = std::max(0, static_cast<int>(cx - sp));
                int maxSX = std::min(width - 1, static_cast<int>(cx + sp));
                int minSY = std::max(0, static_cast<int>(cy - sp));
                int maxSY = std::min(height - 1, static_cast<int>(cy + sp));
                
                for (int sy = minSY; sy <= maxSY; ++sy) {
                    for (int sx = minSX; sx <= maxSX; ++sx) {
                        // Spatial distance
                        double sdist = (sx - cx) * (sx - cx) + (sy - cy) * (sy - cy);
                        if (sdist > sp * sp) continue;
                        
                        // Color distance
                        int sidx = sy * width + sx;
                        double cdist = 0;
                        for (int c = 0; c < channels; ++c) {
                            double d = src[sidx * channels + c] - cc[c];
                            cdist += d * d;
                        }
                        if (cdist > sr * sr) continue;
                        
                        sumX += sx;
                        sumY += sy;
                        for (int c = 0; c < channels; ++c) {
                            sumC[c] += src[sidx * channels + c];
                        }
                        count++;
                    }
                }
                
                if (count > 0) {
                    double newX = sumX / count;
                    double newY = sumY / count;
                    
                    double shift = (newX - cx) * (newX - cx) + (newY - cy) * (newY - cy);
                    
                    cx = newX;
                    cy = newY;
                    for (int c = 0; c < channels; ++c) {
                        cc[c] = sumC[c] / count;
                    }
                    
                    if (shift < epsilon) break;
                }
            }
            
            // Store result
            for (int c = 0; c < channels; ++c) {
                result[idx * channels + c] = static_cast<float>(cc[c]);
            }
        }
    }
    
    std::copy(result.begin(), result.end(), dst);
}

/**
 * @brief K-means based segmentation
 */
inline void kmeansClustering(
    const float* image, int width, int height, int channels,
    int* labels,
    int K,
    int maxIter = 10
) {
    int N = width * height;
    
    // Initialize centers randomly (evenly spaced)
    std::vector<std::vector<float>> centers(K, std::vector<float>(channels));
    for (int k = 0; k < K; ++k) {
        int idx = (k * N) / K;
        for (int c = 0; c < channels; ++c) {
            centers[k][c] = image[idx * channels + c];
        }
    }
    
    // Iterate
    for (int iter = 0; iter < maxIter; ++iter) {
        // Assignment step
        for (int i = 0; i < N; ++i) {
            float minDist = std::numeric_limits<float>::max();
            int bestK = 0;
            
            for (int k = 0; k < K; ++k) {
                float dist = 0;
                for (int c = 0; c < channels; ++c) {
                    float d = image[i * channels + c] - centers[k][c];
                    dist += d * d;
                }
                if (dist < minDist) {
                    minDist = dist;
                    bestK = k;
                }
            }
            labels[i] = bestK;
        }
        
        // Update step
        std::vector<std::vector<float>> newCenters(K, std::vector<float>(channels, 0));
        std::vector<int> counts(K, 0);
        
        for (int i = 0; i < N; ++i) {
            int k = labels[i];
            for (int c = 0; c < channels; ++c) {
                newCenters[k][c] += image[i * channels + c];
            }
            counts[k]++;
        }
        
        bool converged = true;
        for (int k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                for (int c = 0; c < channels; ++c) {
                    float newVal = newCenters[k][c] / counts[k];
                    if (std::abs(newVal - centers[k][c]) > 0.1f) {
                        converged = false;
                    }
                    centers[k][c] = newVal;
                }
            }
        }
        
        if (converged) break;
    }
}

/**
 * @brief SLIC superpixel segmentation (simplified)
 */
inline void slic(
    const float* image, int width, int height, int channels,
    int* labels,
    int numSuperpixels = 100,
    float compactness = 10.0f,
    int maxIter = 10
) {
    // Calculate grid step
    int N = width * height;
    float step = std::sqrt(static_cast<float>(N) / numSuperpixels);
    int gridStep = static_cast<int>(step);
    
    // Initialize centers
    struct Center {
        float x, y;
        std::vector<float> color;
    };
    
    std::vector<Center> centers;
    for (int y = gridStep / 2; y < height; y += gridStep) {
        for (int x = gridStep / 2; x < width; x += gridStep) {
            Center c;
            c.x = static_cast<float>(x);
            c.y = static_cast<float>(y);
            c.color.resize(channels);
            int idx = y * width + x;
            for (int ch = 0; ch < channels; ++ch) {
                c.color[ch] = image[idx * channels + ch];
            }
            centers.push_back(c);
        }
    }
    
    int K = static_cast<int>(centers.size());
    std::vector<float> distances(N, std::numeric_limits<float>::max());
    
    // Iterate
    float S = step;
    float m = compactness;
    
    for (int iter = 0; iter < maxIter; ++iter) {
        // Assignment step
        std::fill(distances.begin(), distances.end(), std::numeric_limits<float>::max());
        
        for (int k = 0; k < K; ++k) {
            int cx = static_cast<int>(centers[k].x);
            int cy = static_cast<int>(centers[k].y);
            
            int minX = std::max(0, cx - static_cast<int>(S));
            int maxX = std::min(width - 1, cx + static_cast<int>(S));
            int minY = std::max(0, cy - static_cast<int>(S));
            int maxY = std::min(height - 1, cy + static_cast<int>(S));
            
            for (int y = minY; y <= maxY; ++y) {
                for (int x = minX; x <= maxX; ++x) {
                    int idx = y * width + x;
                    
                    // Color distance
                    float dc = 0;
                    for (int ch = 0; ch < channels; ++ch) {
                        float d = image[idx * channels + ch] - centers[k].color[ch];
                        dc += d * d;
                    }
                    dc = std::sqrt(dc);
                    
                    // Spatial distance
                    float ds = std::sqrt((x - centers[k].x) * (x - centers[k].x) + 
                                        (y - centers[k].y) * (y - centers[k].y));
                    
                    // Combined distance
                    float D = std::sqrt(dc * dc + (ds / S) * (ds / S) * m * m);
                    
                    if (D < distances[idx]) {
                        distances[idx] = D;
                        labels[idx] = k;
                    }
                }
            }
        }
        
        // Update step
        std::vector<float> sumX(K, 0), sumY(K, 0);
        std::vector<std::vector<float>> sumColor(K, std::vector<float>(channels, 0));
        std::vector<int> counts(K, 0);
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                int k = labels[idx];
                sumX[k] += x;
                sumY[k] += y;
                for (int ch = 0; ch < channels; ++ch) {
                    sumColor[k][ch] += image[idx * channels + ch];
                }
                counts[k]++;
            }
        }
        
        for (int k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                centers[k].x = sumX[k] / counts[k];
                centers[k].y = sumY[k] / counts[k];
                for (int ch = 0; ch < channels; ++ch) {
                    centers[k].color[ch] = sumColor[k][ch] / counts[k];
                }
            }
        }
    }
}

} // namespace imgproc
} // namespace neurova

#endif // NEUROVA_IMGPROC_SEGMENTATION_HPP
