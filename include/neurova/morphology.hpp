// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file morphology.hpp
 * @brief Morphological operations for image processing
 * 
 * Neurova implementation of morphological transforms.
 */

#pragma once

#include "core/image.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace neurova {
namespace morphology {

/**
 * @brief Structuring element shapes
 */
enum class Shape {
    Rectangle,
    Cross,
    Ellipse,
    Diamond
};

/**
 * @brief Create structuring element
 */
inline std::vector<std::vector<uint8_t>> get_structuring_element(Shape shape, int size) {
    std::vector<std::vector<uint8_t>> kernel(size, std::vector<uint8_t>(size, 0));
    int center = size / 2;
    
    switch (shape) {
        case Shape::Rectangle:
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    kernel[i][j] = 1;
                }
            }
            break;
            
        case Shape::Cross:
            for (int i = 0; i < size; ++i) {
                kernel[center][i] = 1;
                kernel[i][center] = 1;
            }
            break;
            
        case Shape::Ellipse:
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    float di = static_cast<float>(i - center) / center;
                    float dj = static_cast<float>(j - center) / center;
                    if (di * di + dj * dj <= 1.0f) {
                        kernel[i][j] = 1;
                    }
                }
            }
            break;
            
        case Shape::Diamond:
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    if (std::abs(i - center) + std::abs(j - center) <= center) {
                        kernel[i][j] = 1;
                    }
                }
            }
            break;
    }
    
    return kernel;
}

/**
 * @brief Create custom structuring element
 */
inline std::vector<std::vector<uint8_t>> get_structuring_element(
    const std::vector<std::vector<uint8_t>>& custom) {
    return custom;
}

/**
 * @brief Erosion operation
 */
inline Image erode(const Image& image, const std::vector<std::vector<uint8_t>>& kernel,
                   int iterations = 1) {
    Image result = image;
    int kh = static_cast<int>(kernel.size());
    int kw = static_cast<int>(kernel[0].size());
    int kcy = kh / 2;
    int kcx = kw / 2;
    
    for (int iter = 0; iter < iterations; ++iter) {
        Image temp(result.width(), result.height(), result.channels());
        
        for (int y = 0; y < result.height(); ++y) {
            for (int x = 0; x < result.width(); ++x) {
                for (int c = 0; c < result.channels(); ++c) {
                    float min_val = 255.0f;
                    
                    for (int ky = 0; ky < kh; ++ky) {
                        for (int kx = 0; kx < kw; ++kx) {
                            if (kernel[ky][kx] == 0) continue;
                            
                            int ny = y + ky - kcy;
                            int nx = x + kx - kcx;
                            
                            if (ny >= 0 && ny < result.height() && 
                                nx >= 0 && nx < result.width()) {
                                float val = result.at(nx, ny, c);
                                min_val = std::min(min_val, val);
                            } else {
                                min_val = 0.0f;
                            }
                        }
                    }
                    
                    temp.at(x, y, c) = min_val;
                }
            }
        }
        
        result = temp;
    }
    
    return result;
}

/**
 * @brief Dilation operation
 */
inline Image dilate(const Image& image, const std::vector<std::vector<uint8_t>>& kernel,
                    int iterations = 1) {
    Image result = image;
    int kh = static_cast<int>(kernel.size());
    int kw = static_cast<int>(kernel[0].size());
    int kcy = kh / 2;
    int kcx = kw / 2;
    
    for (int iter = 0; iter < iterations; ++iter) {
        Image temp(result.width(), result.height(), result.channels());
        
        for (int y = 0; y < result.height(); ++y) {
            for (int x = 0; x < result.width(); ++x) {
                for (int c = 0; c < result.channels(); ++c) {
                    float max_val = 0.0f;
                    
                    for (int ky = 0; ky < kh; ++ky) {
                        for (int kx = 0; kx < kw; ++kx) {
                            if (kernel[ky][kx] == 0) continue;
                            
                            int ny = y + ky - kcy;
                            int nx = x + kx - kcx;
                            
                            if (ny >= 0 && ny < result.height() && 
                                nx >= 0 && nx < result.width()) {
                                float val = result.at(nx, ny, c);
                                max_val = std::max(max_val, val);
                            }
                        }
                    }
                    
                    temp.at(x, y, c) = max_val;
                }
            }
        }
        
        result = temp;
    }
    
    return result;
}

/**
 * @brief Opening operation (erosion followed by dilation)
 */
inline Image opening(const Image& image, const std::vector<std::vector<uint8_t>>& kernel,
                     int iterations = 1) {
    return dilate(erode(image, kernel, iterations), kernel, iterations);
}

/**
 * @brief Closing operation (dilation followed by erosion)
 */
inline Image closing(const Image& image, const std::vector<std::vector<uint8_t>>& kernel,
                     int iterations = 1) {
    return erode(dilate(image, kernel, iterations), kernel, iterations);
}

/**
 * @brief Morphological gradient (dilation - erosion)
 */
inline Image morphological_gradient(const Image& image, 
                                    const std::vector<std::vector<uint8_t>>& kernel) {
    Image dilated = dilate(image, kernel);
    Image eroded = erode(image, kernel);
    
    Image result(image.width(), image.height(), image.channels());
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                result.at(x, y, c) = dilated.at(x, y, c) - eroded.at(x, y, c);
            }
        }
    }
    
    return result;
}

/**
 * @brief Top-hat transform (image - opening)
 */
inline Image top_hat(const Image& image, const std::vector<std::vector<uint8_t>>& kernel) {
    Image opened = opening(image, kernel);
    
    Image result(image.width(), image.height(), image.channels());
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                result.at(x, y, c) = std::max(0.0f, image.at(x, y, c) - opened.at(x, y, c));
            }
        }
    }
    
    return result;
}

/**
 * @brief Black-hat transform (closing - image)
 */
inline Image black_hat(const Image& image, const std::vector<std::vector<uint8_t>>& kernel) {
    Image closed = closing(image, kernel);
    
    Image result(image.width(), image.height(), image.channels());
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                result.at(x, y, c) = std::max(0.0f, closed.at(x, y, c) - image.at(x, y, c));
            }
        }
    }
    
    return result;
}

/**
 * @brief Hit-or-miss transform for pattern detection
 */
inline Image hit_or_miss(const Image& image, 
                         const std::vector<std::vector<uint8_t>>& kernel1,
                         const std::vector<std::vector<uint8_t>>& kernel2) {
    // kernel1: foreground pattern
    // kernel2: background pattern
    
    Image eroded1 = erode(image, kernel1);
    
    // Complement of image
    Image complement(image.width(), image.height(), image.channels());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                complement.at(x, y, c) = 255.0f - image.at(x, y, c);
            }
        }
    }
    
    Image eroded2 = erode(complement, kernel2);
    
    // Intersection
    Image result(image.width(), image.height(), image.channels());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                result.at(x, y, c) = std::min(eroded1.at(x, y, c), eroded2.at(x, y, c));
            }
        }
    }
    
    return result;
}

/**
 * @brief Skeletonization (medial axis transform)
 */
inline Image skeletonize(const Image& image, int max_iterations = 100) {
    auto kernel = get_structuring_element(Shape::Cross, 3);
    Image skeleton(image.width(), image.height(), image.channels());
    Image temp = image;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        Image eroded = erode(temp, kernel);
        Image opened = opening(temp, kernel);
        
        // Add (temp - opened) to skeleton
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x) {
                for (int c = 0; c < image.channels(); ++c) {
                    float diff = temp.at(x, y, c) - opened.at(x, y, c);
                    skeleton.at(x, y, c) = std::max(skeleton.at(x, y, c), diff);
                }
            }
        }
        
        temp = eroded;
        
        // Check if empty
        bool is_empty = true;
        for (int y = 0; y < image.height() && is_empty; ++y) {
            for (int x = 0; x < image.width() && is_empty; ++x) {
                if (temp.at(x, y, 0) > 0) {
                    is_empty = false;
                }
            }
        }
        
        if (is_empty) break;
    }
    
    return skeleton;
}

/**
 * @brief Thinning operation
 */
inline Image thin(const Image& image, int max_iterations = 100) {
    // Zhang-Suen thinning algorithm (simplified)
    Image result = image;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        bool changed = false;
        Image temp = result;
        
        // First pass
        for (int y = 1; y < image.height() - 1; ++y) {
            for (int x = 1; x < image.width() - 1; ++x) {
                if (result.at(x, y, 0) == 0) continue;
                
                // Get 8 neighbors
                int p2 = result.at(x, y-1, 0) > 127 ? 1 : 0;
                int p3 = result.at(x+1, y-1, 0) > 127 ? 1 : 0;
                int p4 = result.at(x+1, y, 0) > 127 ? 1 : 0;
                int p5 = result.at(x+1, y+1, 0) > 127 ? 1 : 0;
                int p6 = result.at(x, y+1, 0) > 127 ? 1 : 0;
                int p7 = result.at(x-1, y+1, 0) > 127 ? 1 : 0;
                int p8 = result.at(x-1, y, 0) > 127 ? 1 : 0;
                int p9 = result.at(x-1, y-1, 0) > 127 ? 1 : 0;
                
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                        (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                
                if (A == 1 && B >= 2 && B <= 6 && 
                    p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0) {
                    temp.at(x, y, 0) = 0;
                    changed = true;
                }
            }
        }
        
        result = temp;
        
        // Second pass
        for (int y = 1; y < image.height() - 1; ++y) {
            for (int x = 1; x < image.width() - 1; ++x) {
                if (result.at(x, y, 0) == 0) continue;
                
                int p2 = result.at(x, y-1, 0) > 127 ? 1 : 0;
                int p3 = result.at(x+1, y-1, 0) > 127 ? 1 : 0;
                int p4 = result.at(x+1, y, 0) > 127 ? 1 : 0;
                int p5 = result.at(x+1, y+1, 0) > 127 ? 1 : 0;
                int p6 = result.at(x, y+1, 0) > 127 ? 1 : 0;
                int p7 = result.at(x-1, y+1, 0) > 127 ? 1 : 0;
                int p8 = result.at(x-1, y, 0) > 127 ? 1 : 0;
                int p9 = result.at(x-1, y-1, 0) > 127 ? 1 : 0;
                
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                        (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                
                if (A == 1 && B >= 2 && B <= 6 && 
                    p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0) {
                    temp.at(x, y, 0) = 0;
                    changed = true;
                }
            }
        }
        
        result = temp;
        
        if (!changed) break;
    }
    
    return result;
}

/**
 * @brief Thickening operation (dual of thinning)
 */
inline Image thicken(const Image& image, int iterations = 1) {
    // Complement -> thin -> complement
    Image complement(image.width(), image.height(), image.channels());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                complement.at(x, y, c) = 255.0f - image.at(x, y, c);
            }
        }
    }
    
    Image thinned = thin(complement, iterations);
    
    Image result(image.width(), image.height(), image.channels());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                result.at(x, y, c) = 255.0f - thinned.at(x, y, c);
            }
        }
    }
    
    return result;
}

/**
 * @brief Binary fill holes
 */
inline Image fill_holes(const Image& image) {
    // Flood fill from edges
    Image result = image;
    Image visited(image.width(), image.height(), 1);
    
    std::vector<std::pair<int, int>> queue;
    
    // Add edge pixels to queue
    for (int x = 0; x < image.width(); ++x) {
        if (image.at(x, 0, 0) < 127) queue.push_back({x, 0});
        if (image.at(x, image.height()-1, 0) < 127) queue.push_back({x, image.height()-1});
    }
    for (int y = 0; y < image.height(); ++y) {
        if (image.at(0, y, 0) < 127) queue.push_back({0, y});
        if (image.at(image.width()-1, y, 0) < 127) queue.push_back({image.width()-1, y});
    }
    
    // Flood fill
    while (!queue.empty()) {
        auto [x, y] = queue.back();
        queue.pop_back();
        
        if (x < 0 || x >= image.width() || y < 0 || y >= image.height()) continue;
        if (visited.at(x, y, 0) > 0) continue;
        if (image.at(x, y, 0) > 127) continue;
        
        visited.at(x, y, 0) = 255;
        
        queue.push_back({x-1, y});
        queue.push_back({x+1, y});
        queue.push_back({x, y-1});
        queue.push_back({x, y+1});
    }
    
    // Fill unvisited regions
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (visited.at(x, y, 0) == 0) {
                for (int c = 0; c < image.channels(); ++c) {
                    result.at(x, y, c) = 255;
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief Remove small objects (area opening)
 */
inline Image remove_small_objects(const Image& image, int min_size) {
    // Connected component labeling
    std::vector<int> labels(image.width() * image.height(), 0);
    int current_label = 0;
    
    auto get_idx = [&](int x, int y) { return y * image.width() + x; };
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (image.at(x, y, 0) < 127 || labels[get_idx(x, y)] > 0) continue;
            
            // BFS to label connected component
            current_label++;
            std::vector<std::pair<int, int>> queue = {{x, y}};
            int component_size = 0;
            std::vector<std::pair<int, int>> component_pixels;
            
            while (!queue.empty()) {
                auto [cx, cy] = queue.back();
                queue.pop_back();
                
                if (cx < 0 || cx >= image.width() || cy < 0 || cy >= image.height()) continue;
                if (image.at(cx, cy, 0) < 127 || labels[get_idx(cx, cy)] > 0) continue;
                
                labels[get_idx(cx, cy)] = current_label;
                component_size++;
                component_pixels.push_back({cx, cy});
                
                queue.push_back({cx-1, cy});
                queue.push_back({cx+1, cy});
                queue.push_back({cx, cy-1});
                queue.push_back({cx, cy+1});
            }
            
            // Mark small components for removal
            if (component_size < min_size) {
                for (auto [px, py] : component_pixels) {
                    labels[get_idx(px, py)] = -1;
                }
            }
        }
    }
    
    // Create result
    Image result = image;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (labels[get_idx(x, y)] == -1) {
                for (int c = 0; c < image.channels(); ++c) {
                    result.at(x, y, c) = 0;
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief Remove small holes (area closing)
 */
inline Image remove_small_holes(const Image& image, int min_size) {
    // Complement -> remove_small_objects -> complement
    Image complement(image.width(), image.height(), image.channels());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                complement.at(x, y, c) = 255.0f - image.at(x, y, c);
            }
        }
    }
    
    Image removed = remove_small_objects(complement, min_size);
    
    Image result(image.width(), image.height(), image.channels());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                result.at(x, y, c) = 255.0f - removed.at(x, y, c);
            }
        }
    }
    
    return result;
}

/**
 * @brief Convex hull of binary image
 */
inline Image convex_hull(const Image& image) {
    // Find contour points
    std::vector<std::pair<int, int>> points;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (image.at(x, y, 0) > 127) {
                points.push_back({x, y});
            }
        }
    }
    
    if (points.size() < 3) return image;
    
    // Graham scan for convex hull
    auto cross = [](const std::pair<int,int>& o, const std::pair<int,int>& a, 
                    const std::pair<int,int>& b) {
        return (a.first - o.first) * (b.second - o.second) - 
               (a.second - o.second) * (b.first - o.first);
    };
    
    // Sort points
    std::sort(points.begin(), points.end());
    
    // Build lower hull
    std::vector<std::pair<int, int>> hull;
    for (const auto& p : points) {
        while (hull.size() >= 2 && cross(hull[hull.size()-2], hull[hull.size()-1], p) <= 0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }
    
    // Build upper hull
    size_t lower_size = hull.size();
    for (auto it = points.rbegin(); it != points.rend(); ++it) {
        while (hull.size() > lower_size && cross(hull[hull.size()-2], hull[hull.size()-1], *it) <= 0) {
            hull.pop_back();
        }
        hull.push_back(*it);
    }
    hull.pop_back();
    
    // Fill convex hull
    Image result(image.width(), image.height(), image.channels());
    
    // Scan line fill
    for (int y = 0; y < image.height(); ++y) {
        std::vector<int> intersections;
        
        for (size_t i = 0; i < hull.size(); ++i) {
            size_t j = (i + 1) % hull.size();
            auto& p1 = hull[i];
            auto& p2 = hull[j];
            
            if ((p1.second <= y && p2.second > y) || (p2.second <= y && p1.second > y)) {
                int x = p1.first + (y - p1.second) * (p2.first - p1.first) / (p2.second - p1.second);
                intersections.push_back(x);
            }
        }
        
        std::sort(intersections.begin(), intersections.end());
        
        for (size_t i = 0; i + 1 < intersections.size(); i += 2) {
            for (int x = intersections[i]; x <= intersections[i+1]; ++x) {
                if (x >= 0 && x < image.width()) {
                    for (int c = 0; c < image.channels(); ++c) {
                        result.at(x, y, c) = 255;
                    }
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief Distance transform
 */
inline Image distance_transform(const Image& image, const std::string& metric = "euclidean") {
    Image result(image.width(), image.height(), 1);
    int inf = image.width() + image.height();
    
    // Initialize
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (image.at(x, y, 0) > 127) {
                result.at(x, y, 0) = 0;
            } else {
                result.at(x, y, 0) = static_cast<float>(inf);
            }
        }
    }
    
    // Forward pass
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (x > 0) {
                result.at(x, y, 0) = std::min(result.at(x, y, 0), result.at(x-1, y, 0) + 1);
            }
            if (y > 0) {
                result.at(x, y, 0) = std::min(result.at(x, y, 0), result.at(x, y-1, 0) + 1);
            }
        }
    }
    
    // Backward pass
    for (int y = image.height() - 1; y >= 0; --y) {
        for (int x = image.width() - 1; x >= 0; --x) {
            if (x < image.width() - 1) {
                result.at(x, y, 0) = std::min(result.at(x, y, 0), result.at(x+1, y, 0) + 1);
            }
            if (y < image.height() - 1) {
                result.at(x, y, 0) = std::min(result.at(x, y, 0), result.at(x, y+1, 0) + 1);
            }
        }
    }
    
    if (metric == "euclidean") {
        // Approximate Euclidean distance
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x) {
                result.at(x, y, 0) = std::sqrt(result.at(x, y, 0));
            }
        }
    }
    
    return result;
}

} // namespace morphology
} // namespace neurova
