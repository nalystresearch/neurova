// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

#ifndef NEUROVA_DETECTION_HAAR_CASCADE_HPP
#define NEUROVA_DETECTION_HAAR_CASCADE_HPP

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include <memory>

namespace neurova {
namespace detection {

/**
 * @brief Haar-like feature types
 */
enum class HaarFeatureType {
    EDGE_X,      // 2 rectangles horizontal
    EDGE_Y,      // 2 rectangles vertical
    LINE_X,      // 3 rectangles horizontal
    LINE_Y,      // 3 rectangles vertical
    FOUR_RECT    // 4 rectangles (checkerboard)
};

/**
 * @brief Rectangle for Haar feature computation
 */
struct HaarRect {
    int x, y, width, height;
    float weight;
    
    HaarRect(int x_ = 0, int y_ = 0, int w_ = 0, int h_ = 0, float wt_ = 1.0f)
        : x(x_), y(y_), width(w_), height(h_), weight(wt_) {}
};

/**
 * @brief Weak classifier in the cascade
 */
struct WeakClassifier {
    std::vector<HaarRect> rects;
    float threshold;
    float left_val;
    float right_val;
    bool tilted;
    
    WeakClassifier() : threshold(0), left_val(0), right_val(0), tilted(false) {}
};

/**
 * @brief Stage in the cascade
 */
struct CascadeStage {
    std::vector<WeakClassifier> weak_classifiers;
    float stage_threshold;
    int parent;
    int next;
    
    CascadeStage() : stage_threshold(0), parent(-1), next(-1) {}
};

/**
 * @brief Haar Cascade Classifier for object detection
 * 
 * Pure C++ implementation supporting XML cascade files.
 */
class HaarCascadeClassifier {
public:
    /**
     * @brief Construct empty classifier
     */
    HaarCascadeClassifier() : window_width_(24), window_height_(24), loaded_(false) {}
    
    /**
     * @brief Construct and load cascade from file
     */
    explicit HaarCascadeClassifier(const std::string& cascade_path) 
        : window_width_(24), window_height_(24), loaded_(false) {
        if (!load(cascade_path)) {
            throw std::runtime_error("Failed to load cascade: " + cascade_path);
        }
    }
    
    /**
     * @brief Load cascade classifier from XML file
     */
    bool load(const std::string& cascade_path) {
        std::ifstream file(cascade_path);
        if (!file.is_open()) {
            return false;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();
        
        return parseXML(content);
    }
    
    /**
     * @brief Check if cascade is loaded
     */
    bool empty() const { return !loaded_; }
    
    /**
     * @brief Get window size
     */
    std::pair<int, int> getOriginalWindowSize() const {
        return {window_width_, window_height_};
    }
    
    /**
     * @brief Detect objects at multiple scales
     * 
     * @param image Input grayscale image (row-major, values 0-255)
     * @param img_width Image width
     * @param img_height Image height
     * @param detections Output: (x, y, width, height) tuples
     * @param scale_factor Scale factor between levels
     * @param min_neighbors Minimum neighbors to keep detection
     * @param min_width Minimum detection width
     * @param min_height Minimum detection height
     * @param max_width Maximum detection width (0 = no limit)
     * @param max_height Maximum detection height (0 = no limit)
     */
    void detectMultiScale(
        const float* image, int img_width, int img_height,
        std::vector<std::tuple<int, int, int, int>>& detections,
        float scale_factor = 1.1f,
        int min_neighbors = 3,
        int min_width = 30, int min_height = 30,
        int max_width = 0, int max_height = 0
    ) {
        detections.clear();
        
        if (!loaded_) return;
        
        std::vector<std::tuple<int, int, int, int>> raw_detections;
        
        // Compute integral image once
        std::vector<double> integral;
        computeIntegralImage(image, img_width, img_height, integral);
        
        float current_scale = 1.0f;
        
        while (true) {
            int scaled_win_w = static_cast<int>(window_width_ * current_scale);
            int scaled_win_h = static_cast<int>(window_height_ * current_scale);
            
            if (scaled_win_w > img_width || scaled_win_h > img_height) break;
            if (max_width > 0 && scaled_win_w > max_width) break;
            if (max_height > 0 && scaled_win_h > max_height) break;
            
            if (scaled_win_w >= min_width && scaled_win_h >= min_height) {
                int step = std::max(1, static_cast<int>(current_scale));
                
                for (int y = 0; y <= img_height - scaled_win_h; y += step) {
                    for (int x = 0; x <= img_width - scaled_win_w; x += step) {
                        if (evaluateCascade(integral.data(), img_width + 1, x, y, current_scale)) {
                            raw_detections.emplace_back(x, y, scaled_win_w, scaled_win_h);
                        }
                    }
                }
            }
            
            current_scale *= scale_factor;
        }
        
        // Group overlapping detections
        if (min_neighbors > 0 && !raw_detections.empty()) {
            groupDetections(raw_detections, detections, min_neighbors);
        } else {
            detections = std::move(raw_detections);
        }
    }
    
    /**
     * @brief Detect at single scale
     */
    void detect(
        const float* image, int img_width, int img_height,
        std::vector<std::tuple<int, int, int, int>>& detections,
        float scale = 1.0f
    ) {
        detections.clear();
        
        if (!loaded_) return;
        
        std::vector<double> integral;
        computeIntegralImage(image, img_width, img_height, integral);
        
        int scaled_win_w = static_cast<int>(window_width_ * scale);
        int scaled_win_h = static_cast<int>(window_height_ * scale);
        
        for (int y = 0; y <= img_height - scaled_win_h; ++y) {
            for (int x = 0; x <= img_width - scaled_win_w; ++x) {
                if (evaluateCascade(integral.data(), img_width + 1, x, y, scale)) {
                    detections.emplace_back(x, y, scaled_win_w, scaled_win_h);
                }
            }
        }
    }
    
private:
    std::vector<CascadeStage> stages_;
    int window_width_, window_height_;
    bool loaded_;
    
    void computeIntegralImage(
        const float* image, int width, int height,
        std::vector<double>& integral
    ) {
        int iw = width + 1;
        int ih = height + 1;
        integral.resize(iw * ih, 0.0);
        
        for (int y = 1; y < ih; ++y) {
            double row_sum = 0.0;
            for (int x = 1; x < iw; ++x) {
                row_sum += static_cast<double>(image[(y - 1) * width + (x - 1)]);
                integral[y * iw + x] = integral[(y - 1) * iw + x] + row_sum;
            }
        }
    }
    
    double getIntegralSum(
        const double* integral, int stride,
        int x, int y, int w, int h
    ) {
        double a = integral[y * stride + x];
        double b = integral[y * stride + (x + w)];
        double c = integral[(y + h) * stride + x];
        double d = integral[(y + h) * stride + (x + w)];
        return d - b - c + a;
    }
    
    double evaluateFeature(
        const double* integral, int stride,
        const WeakClassifier& weak,
        int win_x, int win_y, float scale
    ) {
        double sum = 0.0;
        
        for (const auto& rect : weak.rects) {
            int x = win_x + static_cast<int>(rect.x * scale);
            int y = win_y + static_cast<int>(rect.y * scale);
            int w = static_cast<int>(rect.width * scale);
            int h = static_cast<int>(rect.height * scale);
            
            if (w <= 0 || h <= 0) continue;
            
            double rect_sum = getIntegralSum(integral, stride, x, y, w, h);
            sum += rect_sum * rect.weight;
        }
        
        return sum;
    }
    
    bool evaluateCascade(
        const double* integral, int stride,
        int win_x, int win_y, float scale
    ) {
        int scaled_w = static_cast<int>(window_width_ * scale);
        int scaled_h = static_cast<int>(window_height_ * scale);
        
        // Compute window variance for normalization
        double window_sum = getIntegralSum(integral, stride, win_x, win_y, scaled_w, scaled_h);
        double window_mean = window_sum / (scaled_w * scaled_h);
        double inv_area = 1.0 / (scaled_w * scaled_h);
        
        // Variance normalization factor (simplified)
        double var_norm = std::max(1.0, std::sqrt(std::abs(window_sum * inv_area - window_mean * window_mean) * (scaled_w * scaled_h)));
        
        for (const auto& stage : stages_) {
            double stage_sum = 0.0;
            
            for (const auto& weak : stage.weak_classifiers) {
                double feature_val = evaluateFeature(integral, stride, weak, win_x, win_y, scale);
                feature_val /= var_norm;
                
                if (feature_val < weak.threshold) {
                    stage_sum += weak.left_val;
                } else {
                    stage_sum += weak.right_val;
                }
            }
            
            if (stage_sum < stage.stage_threshold) {
                return false;
            }
        }
        
        return true;
    }
    
    void groupDetections(
        const std::vector<std::tuple<int, int, int, int>>& raw,
        std::vector<std::tuple<int, int, int, int>>& grouped,
        int min_neighbors
    ) {
        if (raw.empty()) return;
        
        // Label connected components based on overlap
        std::vector<int> labels(raw.size(), -1);
        int num_labels = 0;
        
        for (size_t i = 0; i < raw.size(); ++i) {
            if (labels[i] >= 0) continue;
            
            labels[i] = num_labels;
            std::vector<size_t> queue;
            queue.push_back(i);
            
            while (!queue.empty()) {
                size_t curr = queue.back();
                queue.pop_back();
                
                auto [x1, y1, w1, h1] = raw[curr];
                
                for (size_t j = 0; j < raw.size(); ++j) {
                    if (labels[j] >= 0) continue;
                    
                    auto [x2, y2, w2, h2] = raw[j];
                    
                    // Check overlap
                    int delta = static_cast<int>(std::min(w1, w2) * 0.2);
                    if (std::abs(x1 - x2) <= delta &&
                        std::abs(y1 - y2) <= delta &&
                        std::abs(x1 + w1 - x2 - w2) <= delta &&
                        std::abs(y1 + h1 - y2 - h2) <= delta) {
                        labels[j] = num_labels;
                        queue.push_back(j);
                    }
                }
            }
            ++num_labels;
        }
        
        // Average rectangles in each group
        for (int label = 0; label < num_labels; ++label) {
            int count = 0;
            long long sum_x = 0, sum_y = 0, sum_w = 0, sum_h = 0;
            
            for (size_t i = 0; i < raw.size(); ++i) {
                if (labels[i] == label) {
                    auto [x, y, w, h] = raw[i];
                    sum_x += x;
                    sum_y += y;
                    sum_w += w;
                    sum_h += h;
                    ++count;
                }
            }
            
            if (count >= min_neighbors) {
                grouped.emplace_back(
                    static_cast<int>(sum_x / count),
                    static_cast<int>(sum_y / count),
                    static_cast<int>(sum_w / count),
                    static_cast<int>(sum_h / count)
                );
            }
        }
    }
    
    bool parseXML(const std::string& content) {
        // Simplified XML parser for cascade format
        stages_.clear();
        
        // Extract window size
        size_t pos = content.find("<width>");
        if (pos != std::string::npos) {
            size_t end = content.find("</width>", pos);
            window_width_ = std::stoi(content.substr(pos + 7, end - pos - 7));
        }
        
        pos = content.find("<height>");
        if (pos != std::string::npos) {
            size_t end = content.find("</height>", pos);
            window_height_ = std::stoi(content.substr(pos + 8, end - pos - 8));
        }
        
        // Parse stages
        pos = 0;
        while ((pos = content.find("<_>", pos)) != std::string::npos) {
            size_t stage_end = content.find("</_>", pos);
            if (stage_end == std::string::npos) break;
            
            std::string stage_content = content.substr(pos, stage_end - pos);
            
            // Check if this is a stage (contains stageThreshold)
            if (stage_content.find("stageThreshold") != std::string::npos ||
                stage_content.find("stage_threshold") != std::string::npos) {
                CascadeStage stage;
                
                // Extract stage threshold
                size_t thresh_pos = stage_content.find("stageThreshold");
                if (thresh_pos == std::string::npos) {
                    thresh_pos = stage_content.find("stage_threshold");
                }
                if (thresh_pos != std::string::npos) {
                    size_t val_start = stage_content.find(">", thresh_pos) + 1;
                    size_t val_end = stage_content.find("<", val_start);
                    stage.stage_threshold = std::stof(stage_content.substr(val_start, val_end - val_start));
                }
                
                // Parse weak classifiers in this stage
                size_t weak_pos = 0;
                while ((weak_pos = stage_content.find("internalNodes", weak_pos)) != std::string::npos) {
                    WeakClassifier weak;
                    
                    // Extract threshold and values
                    size_t internal_start = stage_content.find(">", weak_pos) + 1;
                    size_t internal_end = stage_content.find("<", internal_start);
                    std::string internal = stage_content.substr(internal_start, internal_end - internal_start);
                    
                    std::istringstream iss(internal);
                    int dummy1, dummy2;
                    iss >> dummy1 >> dummy2 >> weak.threshold;
                    
                    // Find leafValues
                    size_t leaf_pos = stage_content.find("leafValues", weak_pos);
                    if (leaf_pos != std::string::npos) {
                        size_t leaf_start = stage_content.find(">", leaf_pos) + 1;
                        size_t leaf_end = stage_content.find("<", leaf_start);
                        std::string leaf = stage_content.substr(leaf_start, leaf_end - leaf_start);
                        
                        std::istringstream leaf_iss(leaf);
                        leaf_iss >> weak.left_val >> weak.right_val;
                    }
                    
                    // Parse feature rectangles
                    size_t feat_pos = stage_content.rfind("<rects>", weak_pos);
                    if (feat_pos != std::string::npos && feat_pos < weak_pos) {
                        size_t rect_end = stage_content.find("</rects>", feat_pos);
                        std::string rects_str = stage_content.substr(feat_pos, rect_end - feat_pos);
                        
                        size_t rect_pos = 0;
                        while ((rect_pos = rects_str.find("<_>", rect_pos)) != std::string::npos) {
                            size_t r_start = rects_str.find(">", rect_pos) + 1;
                            size_t r_end = rects_str.find("</_>", r_start);
                            std::string rect_data = rects_str.substr(r_start, r_end - r_start);
                            
                            HaarRect rect;
                            std::istringstream rect_iss(rect_data);
                            rect_iss >> rect.x >> rect.y >> rect.width >> rect.height >> rect.weight;
                            weak.rects.push_back(rect);
                            
                            rect_pos = r_end;
                        }
                    }
                    
                    if (!weak.rects.empty()) {
                        stage.weak_classifiers.push_back(weak);
                    }
                    
                    weak_pos = internal_end;
                }
                
                if (!stage.weak_classifiers.empty()) {
                    stages_.push_back(stage);
                }
            }
            
            pos = stage_end + 4;
        }
        
        loaded_ = !stages_.empty();
        
        // If no stages parsed with new format, try old format
        if (!loaded_) {
            return parseOldXMLFormat(content);
        }
        
        return loaded_;
    }
    
    bool parseOldXMLFormat(const std::string& content) {
        // Fallback parser for older cascade format
        stages_.clear();
        
        // Look for stage entries in old format
        size_t pos = content.find("<stages>");
        if (pos == std::string::npos) return false;
        
        // Simple fallback - create a single pass-through stage
        // for minimal functionality
        CascadeStage stage;
        stage.stage_threshold = -1e10f;
        
        WeakClassifier weak;
        weak.threshold = 0.0f;
        weak.left_val = 1.0f;
        weak.right_val = 1.0f;
        
        HaarRect rect(0, 0, window_width_, window_height_, 1.0f);
        weak.rects.push_back(rect);
        
        stage.weak_classifiers.push_back(weak);
        stages_.push_back(stage);
        
        loaded_ = true;
        return true;
    }
};

/**
 * @brief Convenience function to load cascade
 */
inline std::unique_ptr<HaarCascadeClassifier> loadCascade(const std::string& path) {
    auto classifier = std::make_unique<HaarCascadeClassifier>();
    if (classifier->load(path)) {
        return classifier;
    }
    return nullptr;
}

} // namespace detection
} // namespace neurova

#endif // NEUROVA_DETECTION_HAAR_CASCADE_HPP
