// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file solutions/selfie_segmentation.hpp
 * @brief Selfie/portrait segmentation solution
 * 
 * Neurova implementation of person segmentation for portrait mode effects.
 */

#pragma once

#include "../core/image.hpp"
#include <vector>
#include <cmath>

namespace neurova {
namespace solutions {

/**
 * @brief Segmentation result
 */
struct SegmentationMask {
    Image mask;  // Single channel, 0.0-1.0 confidence
    
    SegmentationMask() = default;
    SegmentationMask(int width, int height) : mask(width, height, 1) {}
    
    int width() const { return mask.width(); }
    int height() const { return mask.height(); }
    
    float at(int x, int y) const { return mask.at(x, y, 0); }
    float& at(int x, int y) { return mask.at(x, y, 0); }
    
    // Threshold mask to binary
    Image to_binary(float threshold = 0.5f) const {
        Image binary(width(), height(), 1);
        for (int y = 0; y < height(); ++y) {
            for (int x = 0; x < width(); ++x) {
                binary.at(x, y, 0) = mask.at(x, y, 0) > threshold ? 255.0f : 0.0f;
            }
        }
        return binary;
    }
};

/**
 * @brief Selfie segmentation configuration
 */
struct SelfieSegmentationConfig {
    enum class ModelSelection {
        General = 0,     // Better for general scenes
        Landscape = 1    // Better for landscape mode
    };
    
    ModelSelection model_selection = ModelSelection::General;
    float threshold = 0.5f;
};

/**
 * @brief Selfie segmentation
 */
class SelfieSegmentation {
public:
    explicit SelfieSegmentation(const SelfieSegmentationConfig& config = {}) : config_(config) {}
    
    /**
     * @brief Segment person from background
     */
    SegmentationMask process(const Image& image) {
        SegmentationMask result(image.width(), image.height());
        
        // Simplified segmentation using color-based approach
        // Real implementation would use neural network
        
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x) {
                float confidence = estimate_foreground(image, x, y);
                result.at(x, y) = confidence;
            }
        }
        
        // Apply smoothing
        smooth_mask(result);
        
        return result;
    }
    
    /**
     * @brief Apply background blur (portrait mode effect)
     */
    Image apply_background_blur(const Image& image, const SegmentationMask& mask,
                                 float blur_amount = 20.0f) {
        // Create blurred background
        Image blurred = gaussian_blur(image, static_cast<int>(blur_amount));
        
        // Composite foreground and blurred background
        Image result(image.width(), image.height(), image.channels());
        
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x) {
                float alpha = mask.at(x, y);
                for (int c = 0; c < image.channels(); ++c) {
                    result.at(x, y, c) = alpha * image.at(x, y, c) + 
                                        (1 - alpha) * blurred.at(x, y, c);
                }
            }
        }
        
        return result;
    }
    
    /**
     * @brief Replace background with a solid color
     */
    Image replace_background_color(const Image& image, const SegmentationMask& mask,
                                    float r, float g, float b) {
        Image result(image.width(), image.height(), image.channels());
        
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x) {
                float alpha = mask.at(x, y);
                if (image.channels() >= 3) {
                    result.at(x, y, 0) = alpha * image.at(x, y, 0) + (1 - alpha) * r;
                    result.at(x, y, 1) = alpha * image.at(x, y, 1) + (1 - alpha) * g;
                    result.at(x, y, 2) = alpha * image.at(x, y, 2) + (1 - alpha) * b;
                } else {
                    result.at(x, y, 0) = alpha * image.at(x, y, 0) + (1 - alpha) * (r + g + b) / 3;
                }
            }
        }
        
        return result;
    }
    
    /**
     * @brief Replace background with another image
     */
    Image replace_background_image(const Image& foreground, const SegmentationMask& mask,
                                    const Image& background) {
        Image result(foreground.width(), foreground.height(), foreground.channels());
        
        // Scale background if needed
        float scale_x = static_cast<float>(background.width()) / foreground.width();
        float scale_y = static_cast<float>(background.height()) / foreground.height();
        
        for (int y = 0; y < foreground.height(); ++y) {
            for (int x = 0; x < foreground.width(); ++x) {
                float alpha = mask.at(x, y);
                
                int bg_x = static_cast<int>(x * scale_x);
                int bg_y = static_cast<int>(y * scale_y);
                bg_x = std::min(bg_x, background.width() - 1);
                bg_y = std::min(bg_y, background.height() - 1);
                
                for (int c = 0; c < foreground.channels(); ++c) {
                    float fg_val = foreground.at(x, y, c);
                    float bg_val = c < background.channels() ? background.at(bg_x, bg_y, c) : 0;
                    result.at(x, y, c) = alpha * fg_val + (1 - alpha) * bg_val;
                }
            }
        }
        
        return result;
    }
    
private:
    SelfieSegmentationConfig config_;
    
    float estimate_foreground(const Image& image, int x, int y) {
        // Simple color-based segmentation (placeholder)
        // Assumes person is in center of frame
        
        float cx = image.width() / 2.0f;
        float cy = image.height() / 2.0f;
        
        // Distance from center
        float dx = (x - cx) / cx;
        float dy = (y - cy) / cy;
        float dist = std::sqrt(dx*dx + dy*dy);
        
        // Gaussian falloff from center
        float spatial_weight = std::exp(-dist * dist * 2);
        
        // Color-based weight (skin color detection)
        float color_weight = 0.5f;
        if (image.channels() >= 3) {
            float r = image.at(x, y, 0);
            float g = image.at(x, y, 1);
            float b = image.at(x, y, 2);
            
            // Simple skin color detection
            if (r > 95 && g > 40 && b > 20 &&
                r > g && r > b &&
                std::abs(r - g) > 15 &&
                r - g > 15) {
                color_weight = 0.9f;
            }
        }
        
        return spatial_weight * 0.5f + color_weight * 0.5f;
    }
    
    void smooth_mask(SegmentationMask& mask) {
        // Gaussian smoothing
        int kernel_size = 5;
        int half = kernel_size / 2;
        
        Image temp = mask.mask;
        
        for (int y = half; y < mask.height() - half; ++y) {
            for (int x = half; x < mask.width() - half; ++x) {
                float sum = 0;
                float weight_sum = 0;
                
                for (int ky = -half; ky <= half; ++ky) {
                    for (int kx = -half; kx <= half; ++kx) {
                        float weight = std::exp(-(kx*kx + ky*ky) / 4.0f);
                        sum += weight * temp.at(x + kx, y + ky, 0);
                        weight_sum += weight;
                    }
                }
                
                mask.at(x, y) = sum / weight_sum;
            }
        }
    }
    
    Image gaussian_blur(const Image& image, int kernel_size) {
        Image result(image.width(), image.height(), image.channels());
        int half = kernel_size / 2;
        float sigma = kernel_size / 6.0f;
        
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x) {
                for (int c = 0; c < image.channels(); ++c) {
                    float sum = 0;
                    float weight_sum = 0;
                    
                    for (int ky = -half; ky <= half; ++ky) {
                        for (int kx = -half; kx <= half; ++kx) {
                            int nx = std::max(0, std::min(image.width() - 1, x + kx));
                            int ny = std::max(0, std::min(image.height() - 1, y + ky));
                            
                            float weight = std::exp(-(kx*kx + ky*ky) / (2 * sigma * sigma));
                            sum += weight * image.at(nx, ny, c);
                            weight_sum += weight;
                        }
                    }
                    
                    result.at(x, y, c) = sum / weight_sum;
                }
            }
        }
        
        return result;
    }
};

} // namespace solutions
} // namespace neurova
