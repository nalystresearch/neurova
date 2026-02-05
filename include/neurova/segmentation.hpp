// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file segmentation.hpp
 * @brief Image segmentation algorithms
 * 
 * Neurova implementation of segmentation methods.
 */

#pragma once

#include "core/image.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <random>

namespace neurova {
namespace segmentation {

// ============================================================================
// Thresholding
// ============================================================================

/**
 * @brief Simple binary thresholding
 */
inline Image threshold(const Image& image, float thresh, float max_val = 255.0f) {
    Image result(image.width(), image.height(), image.channels());
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                result.at(x, y, c) = image.at(x, y, c) > thresh ? max_val : 0.0f;
            }
        }
    }
    
    return result;
}

/**
 * @brief Inverse binary thresholding
 */
inline Image threshold_inv(const Image& image, float thresh, float max_val = 255.0f) {
    Image result(image.width(), image.height(), image.channels());
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                result.at(x, y, c) = image.at(x, y, c) > thresh ? 0.0f : max_val;
            }
        }
    }
    
    return result;
}

/**
 * @brief Otsu's automatic thresholding
 */
inline std::pair<Image, float> threshold_otsu(const Image& image) {
    // Build histogram
    std::vector<int> histogram(256, 0);
    int total = image.width() * image.height();
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            int val = static_cast<int>(std::max(0.0f, std::min(255.0f, image.at(x, y, 0))));
            histogram[val]++;
        }
    }
    
    // Calculate cumulative sums
    float sum = 0;
    for (int i = 0; i < 256; ++i) {
        sum += i * histogram[i];
    }
    
    float sumB = 0;
    int wB = 0;
    int wF = 0;
    float max_var = 0;
    int threshold_val = 0;
    
    for (int t = 0; t < 256; ++t) {
        wB += histogram[t];
        if (wB == 0) continue;
        
        wF = total - wB;
        if (wF == 0) break;
        
        sumB += t * histogram[t];
        
        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;
        
        float between_var = static_cast<float>(wB) * static_cast<float>(wF) * 
                           (mB - mF) * (mB - mF);
        
        if (between_var > max_var) {
            max_var = between_var;
            threshold_val = t;
        }
    }
    
    return {threshold(image, static_cast<float>(threshold_val)), static_cast<float>(threshold_val)};
}

/**
 * @brief Adaptive thresholding
 */
inline Image adaptive_threshold(const Image& image, float max_val, int block_size,
                                float C, const std::string& method = "mean") {
    Image result(image.width(), image.height(), image.channels());
    int half = block_size / 2;
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            float local_val = 0;
            int count = 0;
            
            if (method == "gaussian") {
                // Gaussian weighted mean
                float sum = 0;
                float weight_sum = 0;
                float sigma = block_size / 6.0f;
                
                for (int dy = -half; dy <= half; ++dy) {
                    for (int dx = -half; dx <= half; ++dx) {
                        int ny = std::max(0, std::min(image.height() - 1, y + dy));
                        int nx = std::max(0, std::min(image.width() - 1, x + dx));
                        
                        float weight = std::exp(-(dx*dx + dy*dy) / (2 * sigma * sigma));
                        sum += image.at(nx, ny, 0) * weight;
                        weight_sum += weight;
                    }
                }
                local_val = sum / weight_sum;
            } else {
                // Mean
                for (int dy = -half; dy <= half; ++dy) {
                    for (int dx = -half; dx <= half; ++dx) {
                        int ny = std::max(0, std::min(image.height() - 1, y + dy));
                        int nx = std::max(0, std::min(image.width() - 1, x + dx));
                        local_val += image.at(nx, ny, 0);
                        count++;
                    }
                }
                local_val /= count;
            }
            
            float thresh = local_val - C;
            result.at(x, y, 0) = image.at(x, y, 0) > thresh ? max_val : 0.0f;
        }
    }
    
    return result;
}

/**
 * @brief Multi-level Otsu thresholding
 */
inline std::pair<Image, std::vector<float>> threshold_multiotsu(const Image& image, int classes = 3) {
    // Simplified: use uniform quantization for multiple classes
    std::vector<float> thresholds;
    
    for (int i = 1; i < classes; ++i) {
        thresholds.push_back(255.0f * i / classes);
    }
    
    Image result(image.width(), image.height(), image.channels());
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            float val = image.at(x, y, 0);
            float label = 0;
            
            for (size_t i = 0; i < thresholds.size(); ++i) {
                if (val > thresholds[i]) {
                    label = static_cast<float>(i + 1);
                }
            }
            
            result.at(x, y, 0) = label * 255.0f / (classes - 1);
        }
    }
    
    return {result, thresholds};
}

// ============================================================================
// Connected Components
// ============================================================================

/**
 * @brief Connected component labeling
 */
inline std::pair<Image, int> connected_components(const Image& image, int connectivity = 4) {
    Image labels(image.width(), image.height(), 1);
    int current_label = 0;
    
    std::vector<int> parent;
    parent.push_back(0);  // Background
    
    auto find = [&](int x) {
        while (x != parent[x]) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    
    auto unite = [&](int x, int y) {
        int px = find(x);
        int py = find(y);
        if (px != py) {
            parent[px] = py;
        }
    };
    
    // First pass
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (image.at(x, y, 0) < 127) {
                labels.at(x, y, 0) = 0;
                continue;
            }
            
            std::vector<int> neighbors;
            
            if (x > 0 && labels.at(x-1, y, 0) > 0) {
                neighbors.push_back(static_cast<int>(labels.at(x-1, y, 0)));
            }
            if (y > 0 && labels.at(x, y-1, 0) > 0) {
                neighbors.push_back(static_cast<int>(labels.at(x, y-1, 0)));
            }
            
            if (connectivity == 8) {
                if (x > 0 && y > 0 && labels.at(x-1, y-1, 0) > 0) {
                    neighbors.push_back(static_cast<int>(labels.at(x-1, y-1, 0)));
                }
                if (x < image.width() - 1 && y > 0 && labels.at(x+1, y-1, 0) > 0) {
                    neighbors.push_back(static_cast<int>(labels.at(x+1, y-1, 0)));
                }
            }
            
            if (neighbors.empty()) {
                current_label++;
                parent.push_back(current_label);
                labels.at(x, y, 0) = static_cast<float>(current_label);
            } else {
                int min_label = *std::min_element(neighbors.begin(), neighbors.end());
                labels.at(x, y, 0) = static_cast<float>(min_label);
                
                for (int n : neighbors) {
                    unite(n, min_label);
                }
            }
        }
    }
    
    // Second pass: relabel with consecutive numbers
    std::unordered_map<int, int> label_map;
    int num_labels = 0;
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            int old_label = static_cast<int>(labels.at(x, y, 0));
            if (old_label == 0) continue;
            
            int root = find(old_label);
            if (label_map.find(root) == label_map.end()) {
                label_map[root] = ++num_labels;
            }
            labels.at(x, y, 0) = static_cast<float>(label_map[root]);
        }
    }
    
    return {labels, num_labels};
}

/**
 * @brief Get properties of labeled regions
 */
struct RegionProperties {
    int label;
    int area;
    float centroid_x, centroid_y;
    int bbox_x, bbox_y, bbox_w, bbox_h;
    float perimeter;
    float circularity;
};

inline std::vector<RegionProperties> regionprops(const Image& labels, int num_labels) {
    std::vector<RegionProperties> props(num_labels);
    
    // Initialize
    for (int i = 0; i < num_labels; ++i) {
        props[i].label = i + 1;
        props[i].area = 0;
        props[i].centroid_x = 0;
        props[i].centroid_y = 0;
        props[i].bbox_x = labels.width();
        props[i].bbox_y = labels.height();
        props[i].bbox_w = 0;
        props[i].bbox_h = 0;
        props[i].perimeter = 0;
    }
    
    // Calculate properties
    for (int y = 0; y < labels.height(); ++y) {
        for (int x = 0; x < labels.width(); ++x) {
            int label = static_cast<int>(labels.at(x, y, 0));
            if (label == 0) continue;
            
            int idx = label - 1;
            props[idx].area++;
            props[idx].centroid_x += x;
            props[idx].centroid_y += y;
            
            // Bounding box
            props[idx].bbox_x = std::min(props[idx].bbox_x, x);
            props[idx].bbox_y = std::min(props[idx].bbox_y, y);
            props[idx].bbox_w = std::max(props[idx].bbox_w, x);
            props[idx].bbox_h = std::max(props[idx].bbox_h, y);
            
            // Perimeter (count boundary pixels)
            bool is_boundary = false;
            if (x == 0 || x == labels.width() - 1 || 
                y == 0 || y == labels.height() - 1) {
                is_boundary = true;
            } else {
                if (labels.at(x-1, y, 0) != label ||
                    labels.at(x+1, y, 0) != label ||
                    labels.at(x, y-1, 0) != label ||
                    labels.at(x, y+1, 0) != label) {
                    is_boundary = true;
                }
            }
            if (is_boundary) props[idx].perimeter++;
        }
    }
    
    // Finalize
    for (int i = 0; i < num_labels; ++i) {
        if (props[i].area > 0) {
            props[i].centroid_x /= props[i].area;
            props[i].centroid_y /= props[i].area;
            props[i].bbox_w = props[i].bbox_w - props[i].bbox_x + 1;
            props[i].bbox_h = props[i].bbox_h - props[i].bbox_y + 1;
            
            // Circularity: 4 * pi * area / perimeter^2
            if (props[i].perimeter > 0) {
                props[i].circularity = 4 * 3.14159265f * props[i].area / 
                                       (props[i].perimeter * props[i].perimeter);
            }
        }
    }
    
    return props;
}

// ============================================================================
// Watershed
// ============================================================================

/**
 * @brief Watershed segmentation
 */
inline Image watershed(const Image& image, const Image& markers) {
    Image labels = markers;
    
    // Priority queue: (gradient magnitude, x, y, label)
    auto cmp = [](const std::tuple<float, int, int, int>& a, 
                  const std::tuple<float, int, int, int>& b) {
        return std::get<0>(a) > std::get<0>(b);
    };
    std::priority_queue<std::tuple<float, int, int, int>,
                        std::vector<std::tuple<float, int, int, int>>,
                        decltype(cmp)> pq(cmp);
    
    // Initialize queue with marker boundary pixels
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            int label = static_cast<int>(markers.at(x, y, 0));
            if (label > 0) {
                // Check if boundary
                bool is_boundary = false;
                for (int dy = -1; dy <= 1 && !is_boundary; ++dy) {
                    for (int dx = -1; dx <= 1 && !is_boundary; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int ny = y + dy;
                        int nx = x + dx;
                        if (ny >= 0 && ny < image.height() && 
                            nx >= 0 && nx < image.width()) {
                            if (markers.at(nx, ny, 0) == 0) {
                                is_boundary = true;
                            }
                        }
                    }
                }
                
                if (is_boundary) {
                    pq.push({image.at(x, y, 0), x, y, label});
                }
            }
        }
    }
    
    // Process queue
    const int dx[] = {-1, 1, 0, 0, -1, -1, 1, 1};
    const int dy[] = {0, 0, -1, 1, -1, 1, -1, 1};
    
    while (!pq.empty()) {
        auto [grad, x, y, label] = pq.top();
        pq.pop();
        
        // Check neighbors
        for (int i = 0; i < 8; ++i) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            
            if (nx < 0 || nx >= image.width() || ny < 0 || ny >= image.height()) continue;
            if (labels.at(nx, ny, 0) != 0) continue;
            
            // Check if touching multiple labels (watershed line)
            std::set<int> neighbor_labels;
            for (int j = 0; j < 8; ++j) {
                int nnx = nx + dx[j];
                int nny = ny + dy[j];
                if (nnx >= 0 && nnx < image.width() && nny >= 0 && nny < image.height()) {
                    int nl = static_cast<int>(labels.at(nnx, nny, 0));
                    if (nl > 0) neighbor_labels.insert(nl);
                }
            }
            
            if (neighbor_labels.size() == 1) {
                // Assign label
                labels.at(nx, ny, 0) = static_cast<float>(label);
                pq.push({image.at(nx, ny, 0), nx, ny, label});
            } else if (neighbor_labels.size() > 1) {
                // Watershed line
                labels.at(nx, ny, 0) = -1;
            }
        }
    }
    
    return labels;
}

/**
 * @brief Marker-controlled watershed with automatic marker generation
 */
inline Image watershed_auto(const Image& gradient, float h_minima = 10.0f) {
    // Find local minima as markers
    Image markers(gradient.width(), gradient.height(), 1);
    int label = 0;
    
    for (int y = 1; y < gradient.height() - 1; ++y) {
        for (int x = 1; x < gradient.width() - 1; ++x) {
            float val = gradient.at(x, y, 0);
            bool is_minima = true;
            
            for (int dy = -1; dy <= 1 && is_minima; ++dy) {
                for (int dx = -1; dx <= 1 && is_minima; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (gradient.at(x + dx, y + dy, 0) < val - h_minima) {
                        is_minima = false;
                    }
                }
            }
            
            if (is_minima) {
                markers.at(x, y, 0) = static_cast<float>(++label);
            }
        }
    }
    
    return watershed(gradient, markers);
}

// ============================================================================
// Superpixels
// ============================================================================

/**
 * @brief SLIC superpixels (Simple Linear Iterative Clustering)
 */
inline Image slic(const Image& image, int n_segments = 100, float compactness = 10.0f,
                  int max_iter = 10) {
    int height = image.height();
    int width = image.width();
    
    // Calculate grid size
    int S = static_cast<int>(std::sqrt(static_cast<float>(width * height) / n_segments));
    
    // Initialize cluster centers
    struct Cluster {
        float l, a, b;  // Lab color
        float x, y;     // Position
    };
    
    std::vector<Cluster> clusters;
    
    for (int y = S/2; y < height; y += S) {
        for (int x = S/2; x < width; x += S) {
            Cluster c;
            c.l = image.at(x, y, 0);
            c.a = image.channels() > 1 ? image.at(x, y, 1) : c.l;
            c.b = image.channels() > 2 ? image.at(x, y, 2) : c.l;
            c.x = static_cast<float>(x);
            c.y = static_cast<float>(y);
            clusters.push_back(c);
        }
    }
    
    // Labels and distances
    Image labels(width, height, 1);
    std::vector<float> distances(width * height, std::numeric_limits<float>::max());
    
    // Iterate
    for (int iter = 0; iter < max_iter; ++iter) {
        // Reset distances
        std::fill(distances.begin(), distances.end(), std::numeric_limits<float>::max());
        
        // Assign pixels to nearest cluster
        for (size_t k = 0; k < clusters.size(); ++k) {
            auto& c = clusters[k];
            
            int x_start = std::max(0, static_cast<int>(c.x) - S);
            int x_end = std::min(width, static_cast<int>(c.x) + S);
            int y_start = std::max(0, static_cast<int>(c.y) - S);
            int y_end = std::min(height, static_cast<int>(c.y) + S);
            
            for (int y = y_start; y < y_end; ++y) {
                for (int x = x_start; x < x_end; ++x) {
                    float l = image.at(x, y, 0);
                    float a = image.channels() > 1 ? image.at(x, y, 1) : l;
                    float b = image.channels() > 2 ? image.at(x, y, 2) : l;
                    
                    // Color distance
                    float dc = std::sqrt((l - c.l) * (l - c.l) + 
                                        (a - c.a) * (a - c.a) + 
                                        (b - c.b) * (b - c.b));
                    
                    // Spatial distance
                    float ds = std::sqrt((x - c.x) * (x - c.x) + 
                                        (y - c.y) * (y - c.y));
                    
                    // Combined distance
                    float D = std::sqrt(dc * dc + (ds / S) * (ds / S) * compactness * compactness);
                    
                    int idx = y * width + x;
                    if (D < distances[idx]) {
                        distances[idx] = D;
                        labels.at(x, y, 0) = static_cast<float>(k);
                    }
                }
            }
        }
        
        // Update cluster centers
        std::vector<float> sum_l(clusters.size(), 0);
        std::vector<float> sum_a(clusters.size(), 0);
        std::vector<float> sum_b(clusters.size(), 0);
        std::vector<float> sum_x(clusters.size(), 0);
        std::vector<float> sum_y(clusters.size(), 0);
        std::vector<int> count(clusters.size(), 0);
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int k = static_cast<int>(labels.at(x, y, 0));
                
                sum_l[k] += image.at(x, y, 0);
                sum_a[k] += image.channels() > 1 ? image.at(x, y, 1) : image.at(x, y, 0);
                sum_b[k] += image.channels() > 2 ? image.at(x, y, 2) : image.at(x, y, 0);
                sum_x[k] += x;
                sum_y[k] += y;
                count[k]++;
            }
        }
        
        for (size_t k = 0; k < clusters.size(); ++k) {
            if (count[k] > 0) {
                clusters[k].l = sum_l[k] / count[k];
                clusters[k].a = sum_a[k] / count[k];
                clusters[k].b = sum_b[k] / count[k];
                clusters[k].x = sum_x[k] / count[k];
                clusters[k].y = sum_y[k] / count[k];
            }
        }
    }
    
    return labels;
}

// ============================================================================
// Flood Fill
// ============================================================================

/**
 * @brief Flood fill
 */
inline Image flood_fill(const Image& image, int seed_x, int seed_y, 
                        float new_value, float tolerance = 0.0f) {
    Image result = image;
    float target = image.at(seed_x, seed_y, 0);
    
    std::vector<std::pair<int, int>> queue = {{seed_x, seed_y}};
    std::vector<bool> visited(image.width() * image.height(), false);
    
    while (!queue.empty()) {
        auto [x, y] = queue.back();
        queue.pop_back();
        
        if (x < 0 || x >= image.width() || y < 0 || y >= image.height()) continue;
        if (visited[y * image.width() + x]) continue;
        if (std::abs(result.at(x, y, 0) - target) > tolerance) continue;
        
        visited[y * image.width() + x] = true;
        
        for (int c = 0; c < result.channels(); ++c) {
            result.at(x, y, c) = new_value;
        }
        
        queue.push_back({x - 1, y});
        queue.push_back({x + 1, y});
        queue.push_back({x, y - 1});
        queue.push_back({x, y + 1});
    }
    
    return result;
}

// ============================================================================
// Graph Cut (simplified)
// ============================================================================

/**
 * @brief GrabCut-style foreground extraction (simplified)
 */
inline Image grab_cut(const Image& image, int rect_x, int rect_y, int rect_w, int rect_h,
                      int iterations = 5) {
    // Initialize mask
    Image mask(image.width(), image.height(), 1);
    
    // 0: background, 1: foreground, 2: probable background, 3: probable foreground
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (x >= rect_x && x < rect_x + rect_w && 
                y >= rect_y && y < rect_y + rect_h) {
                mask.at(x, y, 0) = 3;  // Probable foreground
            } else {
                mask.at(x, y, 0) = 0;  // Background
            }
        }
    }
    
    // Simple iterative refinement based on color distance
    for (int iter = 0; iter < iterations; ++iter) {
        // Calculate mean foreground color
        float fg_r = 0, fg_g = 0, fg_b = 0;
        int fg_count = 0;
        
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x) {
                if (mask.at(x, y, 0) >= 2) {  // Foreground or probable foreground
                    fg_r += image.at(x, y, 0);
                    if (image.channels() > 1) fg_g += image.at(x, y, 1);
                    if (image.channels() > 2) fg_b += image.at(x, y, 2);
                    fg_count++;
                }
            }
        }
        
        if (fg_count > 0) {
            fg_r /= fg_count;
            fg_g /= fg_count;
            fg_b /= fg_count;
        }
        
        // Calculate mean background color
        float bg_r = 0, bg_g = 0, bg_b = 0;
        int bg_count = 0;
        
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x) {
                if (mask.at(x, y, 0) <= 1) {  // Background or probable background
                    bg_r += image.at(x, y, 0);
                    if (image.channels() > 1) bg_g += image.at(x, y, 1);
                    if (image.channels() > 2) bg_b += image.at(x, y, 2);
                    bg_count++;
                }
            }
        }
        
        if (bg_count > 0) {
            bg_r /= bg_count;
            bg_g /= bg_count;
            bg_b /= bg_count;
        }
        
        // Refine mask based on color distance
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x) {
                if (mask.at(x, y, 0) >= 2) {  // Only update uncertain pixels
                    float r = image.at(x, y, 0);
                    float g = image.channels() > 1 ? image.at(x, y, 1) : r;
                    float b = image.channels() > 2 ? image.at(x, y, 2) : r;
                    
                    float fg_dist = std::sqrt((r - fg_r) * (r - fg_r) + 
                                             (g - fg_g) * (g - fg_g) + 
                                             (b - fg_b) * (b - fg_b));
                    
                    float bg_dist = std::sqrt((r - bg_r) * (r - bg_r) + 
                                             (g - bg_g) * (g - bg_g) + 
                                             (b - bg_b) * (b - bg_b));
                    
                    if (fg_dist < bg_dist) {
                        mask.at(x, y, 0) = 1;  // Foreground
                    } else {
                        mask.at(x, y, 0) = 0;  // Background
                    }
                }
            }
        }
    }
    
    // Convert to binary mask
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            mask.at(x, y, 0) = mask.at(x, y, 0) >= 1 ? 255.0f : 0.0f;
        }
    }
    
    return mask;
}

// ============================================================================
// Contour Finding
// ============================================================================

/**
 * @brief Find contours in binary image
 */
inline std::vector<std::vector<std::pair<int, int>>> find_contours(const Image& image) {
    std::vector<std::vector<std::pair<int, int>>> contours;
    Image visited(image.width(), image.height(), 1);
    
    // Direction vectors for Moore neighborhood tracing
    const int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (image.at(x, y, 0) > 127 && visited.at(x, y, 0) == 0) {
                // Check if it's a boundary pixel
                bool is_boundary = false;
                if (x == 0 || y == 0 || x == image.width() - 1 || y == image.height() - 1) {
                    is_boundary = true;
                } else {
                    for (int d = 0; d < 8; d += 2) {
                        int nx = x + dx[d];
                        int ny = y + dy[d];
                        if (image.at(nx, ny, 0) < 127) {
                            is_boundary = true;
                            break;
                        }
                    }
                }
                
                if (is_boundary) {
                    // Trace contour using Moore-Neighbor tracing
                    std::vector<std::pair<int, int>> contour;
                    int cx = x, cy = y;
                    int start_x = x, start_y = y;
                    int dir = 7;  // Start direction
                    
                    do {
                        contour.push_back({cx, cy});
                        visited.at(cx, cy, 0) = 255;
                        
                        // Find next boundary pixel
                        bool found = false;
                        for (int i = 0; i < 8; ++i) {
                            int d = (dir + 6 + i) % 8;  // Start from dir + 6 (previous - 2)
                            int nx = cx + dx[d];
                            int ny = cy + dy[d];
                            
                            if (nx >= 0 && nx < image.width() && 
                                ny >= 0 && ny < image.height() &&
                                image.at(nx, ny, 0) > 127) {
                                cx = nx;
                                cy = ny;
                                dir = d;
                                found = true;
                                break;
                            }
                        }
                        
                        if (!found) break;
                        
                    } while (cx != start_x || cy != start_y);
                    
                    if (contour.size() > 2) {
                        contours.push_back(contour);
                    }
                }
            }
        }
    }
    
    return contours;
}

/**
 * @brief Draw contours on image
 */
inline Image draw_contours(const Image& image, 
                          const std::vector<std::vector<std::pair<int, int>>>& contours,
                          float color = 255.0f, int thickness = 1) {
    Image result = image;
    
    for (const auto& contour : contours) {
        for (size_t i = 0; i < contour.size(); ++i) {
            auto [x1, y1] = contour[i];
            auto [x2, y2] = contour[(i + 1) % contour.size()];
            
            // Draw line (Bresenham's algorithm)
            int dx = std::abs(x2 - x1);
            int dy = std::abs(y2 - y1);
            int sx = x1 < x2 ? 1 : -1;
            int sy = y1 < y2 ? 1 : -1;
            int err = dx - dy;
            
            while (true) {
                // Draw thick pixel
                for (int ty = -thickness/2; ty <= thickness/2; ++ty) {
                    for (int tx = -thickness/2; tx <= thickness/2; ++tx) {
                        int px = x1 + tx;
                        int py = y1 + ty;
                        if (px >= 0 && px < result.width() && py >= 0 && py < result.height()) {
                            for (int c = 0; c < result.channels(); ++c) {
                                result.at(px, py, c) = color;
                            }
                        }
                    }
                }
                
                if (x1 == x2 && y1 == y2) break;
                
                int e2 = 2 * err;
                if (e2 > -dy) {
                    err -= dy;
                    x1 += sx;
                }
                if (e2 < dx) {
                    err += dx;
                    y1 += sy;
                }
            }
        }
    }
    
    return result;
}

} // namespace segmentation
} // namespace neurova
