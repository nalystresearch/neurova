// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

#ifndef NEUROVA_DETECTION_HOG_HPP
#define NEUROVA_DETECTION_HOG_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace neurova {
namespace detection {

/**
 * @brief Histogram of Oriented Gradients (HOG) descriptor
 * 
 * A feature descriptor used for object detection, especially pedestrian detection.
 */
class HOGDescriptor {
public:
    // Default parameters
    static constexpr int DEFAULT_NLEVELS = 64;
    
    /**
     * @brief Construct HOG descriptor with specified parameters
     */
    HOGDescriptor(
        int win_width = 64, int win_height = 128,
        int block_width = 16, int block_height = 16,
        int block_stride_x = 8, int block_stride_y = 8,
        int cell_width = 8, int cell_height = 8,
        int nbins = 9,
        int deriv_aperture = 1,
        float win_sigma = -1.0f,
        int histogram_norm_type = 0,
        float l2_hys_threshold = 0.2f,
        bool gamma_correction = true,
        int nlevels = 64,
        bool signed_gradient = false
    ) : win_width_(win_width), win_height_(win_height),
        block_width_(block_width), block_height_(block_height),
        block_stride_x_(block_stride_x), block_stride_y_(block_stride_y),
        cell_width_(cell_width), cell_height_(cell_height),
        nbins_(nbins), deriv_aperture_(deriv_aperture),
        histogram_norm_type_(histogram_norm_type),
        l2_hys_threshold_(l2_hys_threshold),
        gamma_correction_(gamma_correction),
        nlevels_(nlevels), signed_gradient_(signed_gradient)
    {
        win_sigma_ = (win_sigma > 0) ? win_sigma : static_cast<float>(block_width + block_height) / 8.0f;
    }
    
    /**
     * @brief Get the size of the HOG descriptor
     */
    int getDescriptorSize() const {
        int cells_per_block_x = block_width_ / cell_width_;
        int cells_per_block_y = block_height_ / cell_height_;
        int blocks_per_win_x = (win_width_ - block_width_) / block_stride_x_ + 1;
        int blocks_per_win_y = (win_height_ - block_height_) / block_stride_y_ + 1;
        return cells_per_block_x * cells_per_block_y * nbins_ * blocks_per_win_x * blocks_per_win_y;
    }
    
    /**
     * @brief Compute HOG descriptors for the image
     * 
     * @param image Input grayscale image (row-major)
     * @param img_width Image width
     * @param img_height Image height
     * @param descriptors Output descriptors
     * @param stride_x Window stride X
     * @param stride_y Window stride Y
     */
    void compute(
        const float* image, int img_width, int img_height,
        std::vector<float>& descriptors,
        int stride_x = 0, int stride_y = 0
    ) {
        if (stride_x == 0) stride_x = block_stride_x_;
        if (stride_y == 0) stride_y = block_stride_y_;
        
        // Apply gamma correction
        std::vector<float> gray(img_width * img_height);
        if (gamma_correction_) {
            for (int i = 0; i < img_width * img_height; ++i) {
                gray[i] = std::sqrt(std::max(0.0f, image[i]));
            }
        } else {
            std::copy(image, image + img_width * img_height, gray.begin());
        }
        
        // Compute gradients
        std::vector<float> gx(img_width * img_height, 0.0f);
        std::vector<float> gy(img_width * img_height, 0.0f);
        
        computeSobel(gray.data(), img_width, img_height, gx.data(), gy.data());
        
        // Compute magnitude and orientation
        std::vector<float> magnitude(img_width * img_height);
        std::vector<float> orientation(img_width * img_height);
        float max_angle = signed_gradient_ ? 360.0f : 180.0f;
        
        for (int i = 0; i < img_width * img_height; ++i) {
            magnitude[i] = std::sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
            float angle = std::atan2(gy[i], gx[i]) * 180.0f / M_PI;
            if (signed_gradient_) {
                if (angle < 0) angle += 360.0f;
            } else {
                angle = std::abs(angle);
                if (angle > 180.0f) angle = 360.0f - angle;
            }
            orientation[i] = angle;
        }
        
        // Compute descriptors for each window
        descriptors.clear();
        
        for (int y = 0; y <= img_height - win_height_; y += stride_y) {
            for (int x = 0; x <= img_width - win_width_; x += stride_x) {
                std::vector<float> window_desc;
                computeWindow(magnitude.data(), orientation.data(),
                             img_width, img_height,
                             x, y, max_angle, window_desc);
                descriptors.insert(descriptors.end(), window_desc.begin(), window_desc.end());
            }
        }
    }
    
    /**
     * @brief Set SVM detector coefficients
     */
    void setSVMDetector(const std::vector<float>& detector) {
        svm_detector_ = detector;
    }
    
    /**
     * @brief Get default people detector coefficients (placeholder)
     */
    static std::vector<float> getDefaultPeopleDetector() {
        // Returns zero-initialized detector (in practice, use trained coefficients)
        return std::vector<float>(3780 + 1, 0.0f);
    }
    
    /**
     * @brief Multi-scale object detection
     * 
     * @param image Input image
     * @param img_width Image width
     * @param img_height Image height
     * @param detections Output detections (x, y, w, h)
     * @param weights Output detection weights
     * @param hit_threshold SVM threshold
     * @param scale Scale factor between levels
     * @param group_threshold Minimum detections to keep
     */
    void detectMultiScale(
        const float* image, int img_width, int img_height,
        std::vector<std::tuple<int, int, int, int>>& detections,
        std::vector<float>& weights,
        float hit_threshold = 0.0f,
        float scale = 1.05f,
        int group_threshold = 2
    ) {
        detections.clear();
        weights.clear();
        
        if (svm_detector_.empty()) return;
        
        float current_scale = 1.0f;
        
        for (int level = 0; level < nlevels_; ++level) {
            int scaled_w = static_cast<int>(img_width / current_scale);
            int scaled_h = static_cast<int>(img_height / current_scale);
            
            if (scaled_w < win_width_ || scaled_h < win_height_) break;
            
            // Resize image
            std::vector<float> scaled_img(scaled_w * scaled_h);
            resizeNearest(image, img_width, img_height,
                         scaled_img.data(), scaled_w, scaled_h);
            
            // Compute descriptors
            std::vector<float> descriptors;
            compute(scaled_img.data(), scaled_w, scaled_h, descriptors);
            
            if (descriptors.empty()) {
                current_scale *= scale;
                continue;
            }
            
            int desc_size = getDescriptorSize();
            int num_windows = descriptors.size() / desc_size;
            
            int grid_w = (scaled_w - win_width_) / block_stride_x_ + 1;
            
            for (int i = 0; i < num_windows; ++i) {
                // Apply SVM
                float score = svm_detector_.back(); // bias
                for (int j = 0; j < desc_size && j < static_cast<int>(svm_detector_.size()) - 1; ++j) {
                    score += svm_detector_[j] * descriptors[i * desc_size + j];
                }
                
                if (score > hit_threshold) {
                    int grid_y = i / grid_w;
                    int grid_x = i % grid_w;
                    int x = static_cast<int>(grid_x * block_stride_x_ * current_scale);
                    int y = static_cast<int>(grid_y * block_stride_y_ * current_scale);
                    int w = static_cast<int>(win_width_ * current_scale);
                    int h = static_cast<int>(win_height_ * current_scale);
                    
                    detections.emplace_back(x, y, w, h);
                    weights.push_back(score);
                }
            }
            
            current_scale *= scale;
        }
        
        // Group overlapping detections
        if (group_threshold > 0 && !detections.empty()) {
            groupRectangles(detections, weights, group_threshold);
        }
    }
    
private:
    int win_width_, win_height_;
    int block_width_, block_height_;
    int block_stride_x_, block_stride_y_;
    int cell_width_, cell_height_;
    int nbins_;
    int deriv_aperture_;
    float win_sigma_;
    int histogram_norm_type_;
    float l2_hys_threshold_;
    bool gamma_correction_;
    int nlevels_;
    bool signed_gradient_;
    std::vector<float> svm_detector_;
    
    void computeSobel(const float* img, int w, int h, float* gx, float* gy) {
        // Sobel kernels
        static const float kx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        static const float ky[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
        
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                float sum_x = 0, sum_y = 0;
                int k = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        float val = img[(y + dy) * w + (x + dx)];
                        sum_x += kx[k] * val;
                        sum_y += ky[k] * val;
                        ++k;
                    }
                }
                gx[y * w + x] = sum_x;
                gy[y * w + x] = sum_y;
            }
        }
    }
    
    void computeWindow(
        const float* magnitude, const float* orientation,
        int img_width, int img_height,
        int win_x, int win_y, float max_angle,
        std::vector<float>& descriptor
    ) {
        int cells_per_block_x = block_width_ / cell_width_;
        int cells_per_block_y = block_height_ / cell_height_;
        float bin_width = max_angle / nbins_;
        
        descriptor.clear();
        
        // Iterate over blocks
        for (int by = 0; by <= win_height_ - block_height_; by += block_stride_y_) {
            for (int bx = 0; bx <= win_width_ - block_width_; bx += block_stride_x_) {
                std::vector<float> block_hist;
                
                // Iterate over cells in block
                for (int cy = 0; cy < cells_per_block_y; ++cy) {
                    for (int cx = 0; cx < cells_per_block_x; ++cx) {
                        int cell_x = win_x + bx + cx * cell_width_;
                        int cell_y = win_y + by + cy * cell_height_;
                        
                        std::vector<float> cell_hist(nbins_, 0.0f);
                        
                        for (int i = 0; i < cell_height_; ++i) {
                            for (int j = 0; j < cell_width_; ++j) {
                                int px = cell_x + j;
                                int py = cell_y + i;
                                if (px >= 0 && px < img_width && py >= 0 && py < img_height) {
                                    float mag = magnitude[py * img_width + px];
                                    float ori = orientation[py * img_width + px];
                                    
                                    // Bilinear interpolation
                                    float bin_idx = ori / bin_width;
                                    int left_bin = static_cast<int>(bin_idx) % nbins_;
                                    int right_bin = (left_bin + 1) % nbins_;
                                    float right_weight = bin_idx - static_cast<int>(bin_idx);
                                    
                                    cell_hist[left_bin] += (1.0f - right_weight) * mag;
                                    cell_hist[right_bin] += right_weight * mag;
                                }
                            }
                        }
                        
                        block_hist.insert(block_hist.end(), cell_hist.begin(), cell_hist.end());
                    }
                }
                
                // L2-Hys normalization
                float norm = 0.0f;
                for (float v : block_hist) norm += v * v;
                norm = std::sqrt(norm + 1e-6f);
                for (float& v : block_hist) v /= norm;
                
                // Clip and renormalize
                for (float& v : block_hist) {
                    v = std::min(v, l2_hys_threshold_);
                }
                norm = 0.0f;
                for (float v : block_hist) norm += v * v;
                norm = std::sqrt(norm + 1e-6f);
                for (float& v : block_hist) v /= norm;
                
                descriptor.insert(descriptor.end(), block_hist.begin(), block_hist.end());
            }
        }
    }
    
    void resizeNearest(const float* src, int src_w, int src_h,
                       float* dst, int dst_w, int dst_h) {
        for (int y = 0; y < dst_h; ++y) {
            int src_y = y * src_h / dst_h;
            src_y = std::min(src_y, src_h - 1);
            for (int x = 0; x < dst_w; ++x) {
                int src_x = x * src_w / dst_w;
                src_x = std::min(src_x, src_w - 1);
                dst[y * dst_w + x] = src[src_y * src_w + src_x];
            }
        }
    }
    
    void groupRectangles(
        std::vector<std::tuple<int, int, int, int>>& rects,
        std::vector<float>& weights,
        int threshold
    ) {
        if (rects.empty()) return;
        
        // Simple NMS-based grouping
        std::vector<int> indices(rects.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&weights](int a, int b) {
            return weights[a] > weights[b];
        });
        
        std::vector<bool> suppressed(rects.size(), false);
        std::vector<std::tuple<int, int, int, int>> result_rects;
        std::vector<float> result_weights;
        
        for (int i : indices) {
            if (suppressed[i]) continue;
            
            result_rects.push_back(rects[i]);
            result_weights.push_back(weights[i]);
            
            auto [x1, y1, w1, h1] = rects[i];
            
            for (int j : indices) {
                if (suppressed[j] || i == j) continue;
                
                auto [x2, y2, w2, h2] = rects[j];
                
                // Compute IoU
                int xi1 = std::max(x1, x2);
                int yi1 = std::max(y1, y2);
                int xi2 = std::min(x1 + w1, x2 + w2);
                int yi2 = std::min(y1 + h1, y2 + h2);
                
                if (xi2 > xi1 && yi2 > yi1) {
                    float intersection = static_cast<float>((xi2 - xi1) * (yi2 - yi1));
                    float union_area = static_cast<float>(w1 * h1 + w2 * h2) - intersection;
                    float iou = intersection / std::max(union_area, 1e-6f);
                    
                    if (iou > 0.3f) {
                        suppressed[j] = true;
                    }
                }
            }
        }
        
        rects = std::move(result_rects);
        weights = std::move(result_weights);
    }
};

/**
 * @brief Group overlapping rectangles
 */
inline void groupRectangles(
    std::vector<std::tuple<int, int, int, int>>& rectList,
    std::vector<int>& weights,
    int groupThreshold,
    float eps = 0.2f
) {
    if (rectList.empty()) {
        weights.clear();
        return;
    }
    
    // Simple grouping based on IoU overlap
    std::vector<bool> used(rectList.size(), false);
    std::vector<std::tuple<int, int, int, int>> result;
    std::vector<int> result_weights;
    
    for (size_t i = 0; i < rectList.size(); ++i) {
        if (used[i]) continue;
        
        std::vector<size_t> group;
        group.push_back(i);
        used[i] = true;
        
        auto [x1, y1, w1, h1] = rectList[i];
        
        for (size_t j = i + 1; j < rectList.size(); ++j) {
            if (used[j]) continue;
            
            auto [x2, y2, w2, h2] = rectList[j];
            
            // Compute IoU
            int xi1 = std::max(x1, x2);
            int yi1 = std::max(y1, y2);
            int xi2 = std::min(x1 + w1, x2 + w2);
            int yi2 = std::min(y1 + h1, y2 + h2);
            
            float iou = 0.0f;
            if (xi2 > xi1 && yi2 > yi1) {
                float intersection = static_cast<float>((xi2 - xi1) * (yi2 - yi1));
                float union_area = static_cast<float>(w1 * h1 + w2 * h2) - intersection;
                iou = intersection / std::max(union_area, 1e-6f);
            }
            
            if (iou > eps) {
                group.push_back(j);
                used[j] = true;
            }
        }
        
        if (static_cast<int>(group.size()) >= groupThreshold) {
            // Average the group
            int avg_x = 0, avg_y = 0, avg_w = 0, avg_h = 0;
            for (size_t idx : group) {
                auto [x, y, w, h] = rectList[idx];
                avg_x += x;
                avg_y += y;
                avg_w += w;
                avg_h += h;
            }
            int n = static_cast<int>(group.size());
            result.emplace_back(avg_x / n, avg_y / n, avg_w / n, avg_h / n);
            result_weights.push_back(n);
        }
    }
    
    rectList = std::move(result);
    weights = std::move(result_weights);
}

} // namespace detection
} // namespace neurova

#endif // NEUROVA_DETECTION_HOG_HPP
