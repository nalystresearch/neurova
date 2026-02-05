// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file augmentation.hpp
 * @brief Image augmentation transforms
 * 
 * Neurova implementation of data augmentation for training.
 */

#pragma once

#include "core/image.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>

namespace neurova {
namespace augmentation {

/**
 * @brief Base class for transforms
 */
class Transform {
public:
    virtual ~Transform() = default;
    virtual Image apply(const Image& image) = 0;
    
    Image operator()(const Image& image) {
        return apply(image);
    }
};

/**
 * @brief Compose multiple transforms
 */
class Compose : public Transform {
private:
    std::vector<std::unique_ptr<Transform>> transforms_;
    
public:
    template<typename... Ts>
    Compose(Ts&&... transforms) {
        (transforms_.push_back(std::make_unique<std::decay_t<Ts>>(std::forward<Ts>(transforms))), ...);
    }
    
    void add(std::unique_ptr<Transform> transform) {
        transforms_.push_back(std::move(transform));
    }
    
    Image apply(const Image& image) override {
        Image result = image;
        for (auto& t : transforms_) {
            result = t->apply(result);
        }
        return result;
    }
};

/**
 * @brief Randomly apply a transform with given probability
 */
class RandomApply : public Transform {
private:
    std::unique_ptr<Transform> transform_;
    float probability_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandomApply(std::unique_ptr<Transform> transform, float probability = 0.5f)
        : transform_(std::move(transform)), probability_(probability) {}
    
    Image apply(const Image& image) override {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(gen_) < probability_) {
            return transform_->apply(image);
        }
        return image;
    }
};

/**
 * @brief Randomly choose one transform from a list
 */
class RandomChoice : public Transform {
private:
    std::vector<std::unique_ptr<Transform>> transforms_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    void add(std::unique_ptr<Transform> transform) {
        transforms_.push_back(std::move(transform));
    }
    
    Image apply(const Image& image) override {
        if (transforms_.empty()) return image;
        std::uniform_int_distribution<size_t> dist(0, transforms_.size() - 1);
        return transforms_[dist(gen_)]->apply(image);
    }
};

// ============================================================================
// Geometric Transforms
// ============================================================================

/**
 * @brief Resize image
 */
class Resize : public Transform {
private:
    int width_;
    int height_;
    std::string interpolation_;
    
public:
    Resize(int size) : width_(size), height_(size), interpolation_("bilinear") {}
    Resize(int width, int height, const std::string& interpolation = "bilinear")
        : width_(width), height_(height), interpolation_(interpolation) {}
    
    Image apply(const Image& image) override {
        return image.resize(width_, height_);
    }
};

/**
 * @brief Random crop
 */
class RandomCrop : public Transform {
private:
    int width_;
    int height_;
    int padding_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandomCrop(int size, int padding = 0) : width_(size), height_(size), padding_(padding) {}
    RandomCrop(int width, int height, int padding = 0)
        : width_(width), height_(height), padding_(padding) {}
    
    Image apply(const Image& image) override {
        Image padded = image;
        if (padding_ > 0) {
            // Apply padding (simplified - just use original)
            padded = image;
        }
        
        int max_x = std::max(0, padded.width() - width_);
        int max_y = std::max(0, padded.height() - height_);
        
        std::uniform_int_distribution<int> dist_x(0, max_x);
        std::uniform_int_distribution<int> dist_y(0, max_y);
        
        int x = dist_x(gen_);
        int y = dist_y(gen_);
        
        return padded.crop(x, y, width_, height_);
    }
};

/**
 * @brief Center crop
 */
class CenterCrop : public Transform {
private:
    int width_;
    int height_;
    
public:
    CenterCrop(int size) : width_(size), height_(size) {}
    CenterCrop(int width, int height) : width_(width), height_(height) {}
    
    Image apply(const Image& image) override {
        int x = (image.width() - width_) / 2;
        int y = (image.height() - height_) / 2;
        return image.crop(x, y, width_, height_);
    }
};

/**
 * @brief Random horizontal flip
 */
class RandomHorizontalFlip : public Transform {
private:
    float probability_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandomHorizontalFlip(float probability = 0.5f) : probability_(probability) {}
    
    Image apply(const Image& image) override {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(gen_) < probability_) {
            return image.flip_horizontal();
        }
        return image;
    }
};

/**
 * @brief Random vertical flip
 */
class RandomVerticalFlip : public Transform {
private:
    float probability_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandomVerticalFlip(float probability = 0.5f) : probability_(probability) {}
    
    Image apply(const Image& image) override {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(gen_) < probability_) {
            return image.flip_vertical();
        }
        return image;
    }
};

/**
 * @brief Random rotation
 */
class RandomRotation : public Transform {
private:
    float min_degrees_;
    float max_degrees_;
    bool expand_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandomRotation(float degrees, bool expand = false)
        : min_degrees_(-degrees), max_degrees_(degrees), expand_(expand) {}
    
    RandomRotation(float min_degrees, float max_degrees, bool expand = false)
        : min_degrees_(min_degrees), max_degrees_(max_degrees), expand_(expand) {}
    
    Image apply(const Image& image) override {
        std::uniform_real_distribution<float> dist(min_degrees_, max_degrees_);
        float angle = dist(gen_);
        return image.rotate(angle);
    }
};

/**
 * @brief Random affine transformation
 */
class RandomAffine : public Transform {
private:
    float degrees_;
    std::pair<float, float> translate_;
    std::pair<float, float> scale_;
    std::pair<float, float> shear_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandomAffine(float degrees = 0.0f,
                 std::pair<float, float> translate = {0.0f, 0.0f},
                 std::pair<float, float> scale = {1.0f, 1.0f},
                 std::pair<float, float> shear = {0.0f, 0.0f})
        : degrees_(degrees), translate_(translate), scale_(scale), shear_(shear) {}
    
    Image apply(const Image& image) override {
        std::uniform_real_distribution<float> angle_dist(-degrees_, degrees_);
        std::uniform_real_distribution<float> tx_dist(-translate_.first, translate_.first);
        std::uniform_real_distribution<float> ty_dist(-translate_.second, translate_.second);
        std::uniform_real_distribution<float> scale_dist(scale_.first, scale_.second);
        std::uniform_real_distribution<float> shear_dist(shear_.first, shear_.second);
        
        float angle = angle_dist(gen_);
        float tx = tx_dist(gen_) * image.width();
        float ty = ty_dist(gen_) * image.height();
        float s = scale_dist(gen_);
        float shear_x = shear_dist(gen_);
        
        // Build affine matrix
        float cos_a = std::cos(angle * 3.14159265f / 180.0f);
        float sin_a = std::sin(angle * 3.14159265f / 180.0f);
        
        std::vector<std::vector<float>> matrix = {
            {s * cos_a, -s * sin_a + shear_x, tx},
            {s * sin_a, s * cos_a, ty}
        };
        
        return image.warp_affine(matrix);
    }
};

/**
 * @brief Random perspective transformation
 */
class RandomPerspective : public Transform {
private:
    float distortion_scale_;
    float probability_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandomPerspective(float distortion_scale = 0.5f, float probability = 0.5f)
        : distortion_scale_(distortion_scale), probability_(probability) {}
    
    Image apply(const Image& image) override {
        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        if (prob_dist(gen_) >= probability_) {
            return image;
        }
        
        int w = image.width();
        int h = image.height();
        float half_h = distortion_scale_ * h / 2.0f;
        float half_w = distortion_scale_ * w / 2.0f;
        
        std::uniform_real_distribution<float> dist_h(-half_h, half_h);
        std::uniform_real_distribution<float> dist_w(-half_w, half_w);
        
        // Generate random corner offsets
        std::vector<std::pair<float, float>> src_points = {
            {0, 0}, {static_cast<float>(w), 0}, 
            {static_cast<float>(w), static_cast<float>(h)}, {0, static_cast<float>(h)}
        };
        
        std::vector<std::pair<float, float>> dst_points = {
            {dist_w(gen_), dist_h(gen_)},
            {w + dist_w(gen_), dist_h(gen_)},
            {w + dist_w(gen_), h + dist_h(gen_)},
            {dist_w(gen_), h + dist_h(gen_)}
        };
        
        return image.warp_perspective(src_points, dst_points);
    }
};

/**
 * @brief Random resized crop
 */
class RandomResizedCrop : public Transform {
private:
    int size_;
    std::pair<float, float> scale_;
    std::pair<float, float> ratio_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandomResizedCrop(int size, std::pair<float, float> scale = {0.08f, 1.0f},
                      std::pair<float, float> ratio = {0.75f, 1.333333f})
        : size_(size), scale_(scale), ratio_(ratio) {}
    
    Image apply(const Image& image) override {
        int w = image.width();
        int h = image.height();
        float area = static_cast<float>(w * h);
        
        std::uniform_real_distribution<float> scale_dist(scale_.first, scale_.second);
        std::uniform_real_distribution<float> ratio_dist(ratio_.first, ratio_.second);
        
        for (int attempt = 0; attempt < 10; ++attempt) {
            float target_area = area * scale_dist(gen_);
            float aspect_ratio = ratio_dist(gen_);
            
            int crop_w = static_cast<int>(std::sqrt(target_area * aspect_ratio));
            int crop_h = static_cast<int>(std::sqrt(target_area / aspect_ratio));
            
            if (crop_w <= w && crop_h <= h) {
                std::uniform_int_distribution<int> x_dist(0, w - crop_w);
                std::uniform_int_distribution<int> y_dist(0, h - crop_h);
                
                int x = x_dist(gen_);
                int y = y_dist(gen_);
                
                return image.crop(x, y, crop_w, crop_h).resize(size_, size_);
            }
        }
        
        // Fallback: center crop
        float ratio = std::min(static_cast<float>(w) / h, static_cast<float>(h) / w);
        int crop_size = static_cast<int>(std::min(w, h) * ratio);
        int x = (w - crop_size) / 2;
        int y = (h - crop_size) / 2;
        return image.crop(x, y, crop_size, crop_size).resize(size_, size_);
    }
};

// ============================================================================
// Color Transforms
// ============================================================================

/**
 * @brief Normalize image with mean and std
 */
class Normalize : public Transform {
private:
    std::vector<float> mean_;
    std::vector<float> std_;
    
public:
    Normalize(const std::vector<float>& mean, const std::vector<float>& std)
        : mean_(mean), std_(std) {}
    
    // ImageNet defaults
    static Normalize imagenet() {
        return Normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f});
    }
    
    Image apply(const Image& image) override {
        Image result = image;
        int channels = image.channels();
        int pixels = image.width() * image.height();
        
        for (int c = 0; c < channels && c < static_cast<int>(mean_.size()); ++c) {
            for (int i = 0; i < pixels; ++i) {
                float& pixel = result.data()[i * channels + c];
                pixel = (pixel / 255.0f - mean_[c]) / std_[c];
            }
        }
        
        return result;
    }
};

/**
 * @brief Random brightness adjustment
 */
class RandomBrightness : public Transform {
private:
    float min_factor_;
    float max_factor_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandomBrightness(float factor)
        : min_factor_(std::max(0.0f, 1.0f - factor)), max_factor_(1.0f + factor) {}
    
    RandomBrightness(float min_factor, float max_factor)
        : min_factor_(min_factor), max_factor_(max_factor) {}
    
    Image apply(const Image& image) override {
        std::uniform_real_distribution<float> dist(min_factor_, max_factor_);
        float factor = dist(gen_);
        return image.adjust_brightness(factor);
    }
};

/**
 * @brief Random contrast adjustment
 */
class RandomContrast : public Transform {
private:
    float min_factor_;
    float max_factor_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandomContrast(float factor)
        : min_factor_(std::max(0.0f, 1.0f - factor)), max_factor_(1.0f + factor) {}
    
    RandomContrast(float min_factor, float max_factor)
        : min_factor_(min_factor), max_factor_(max_factor) {}
    
    Image apply(const Image& image) override {
        std::uniform_real_distribution<float> dist(min_factor_, max_factor_);
        float factor = dist(gen_);
        return image.adjust_contrast(factor);
    }
};

/**
 * @brief Random saturation adjustment
 */
class RandomSaturation : public Transform {
private:
    float min_factor_;
    float max_factor_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandomSaturation(float factor)
        : min_factor_(std::max(0.0f, 1.0f - factor)), max_factor_(1.0f + factor) {}
    
    RandomSaturation(float min_factor, float max_factor)
        : min_factor_(min_factor), max_factor_(max_factor) {}
    
    Image apply(const Image& image) override {
        std::uniform_real_distribution<float> dist(min_factor_, max_factor_);
        float factor = dist(gen_);
        return image.adjust_saturation(factor);
    }
};

/**
 * @brief Random hue adjustment
 */
class RandomHue : public Transform {
private:
    float hue_factor_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandomHue(float hue_factor) : hue_factor_(hue_factor) {}
    
    Image apply(const Image& image) override {
        std::uniform_real_distribution<float> dist(-hue_factor_, hue_factor_);
        float factor = dist(gen_);
        return image.adjust_hue(factor);
    }
};

/**
 * @brief ColorJitter - random brightness, contrast, saturation, hue
 */
class ColorJitter : public Transform {
private:
    float brightness_;
    float contrast_;
    float saturation_;
    float hue_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    ColorJitter(float brightness = 0.0f, float contrast = 0.0f,
                float saturation = 0.0f, float hue = 0.0f)
        : brightness_(brightness), contrast_(contrast),
          saturation_(saturation), hue_(hue) {}
    
    Image apply(const Image& image) override {
        Image result = image;
        
        // Randomize order of transforms
        std::vector<int> order = {0, 1, 2, 3};
        std::shuffle(order.begin(), order.end(), gen_);
        
        for (int idx : order) {
            switch (idx) {
                case 0:
                    if (brightness_ > 0) {
                        std::uniform_real_distribution<float> dist(
                            std::max(0.0f, 1.0f - brightness_), 1.0f + brightness_);
                        result = result.adjust_brightness(dist(gen_));
                    }
                    break;
                case 1:
                    if (contrast_ > 0) {
                        std::uniform_real_distribution<float> dist(
                            std::max(0.0f, 1.0f - contrast_), 1.0f + contrast_);
                        result = result.adjust_contrast(dist(gen_));
                    }
                    break;
                case 2:
                    if (saturation_ > 0) {
                        std::uniform_real_distribution<float> dist(
                            std::max(0.0f, 1.0f - saturation_), 1.0f + saturation_);
                        result = result.adjust_saturation(dist(gen_));
                    }
                    break;
                case 3:
                    if (hue_ > 0) {
                        std::uniform_real_distribution<float> dist(-hue_, hue_);
                        result = result.adjust_hue(dist(gen_));
                    }
                    break;
            }
        }
        
        return result;
    }
};

/**
 * @brief Convert to grayscale
 */
class Grayscale : public Transform {
private:
    int num_output_channels_;
    
public:
    Grayscale(int num_output_channels = 1) : num_output_channels_(num_output_channels) {}
    
    Image apply(const Image& image) override {
        return image.to_grayscale(num_output_channels_);
    }
};

/**
 * @brief Random grayscale conversion
 */
class RandomGrayscale : public Transform {
private:
    float probability_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandomGrayscale(float probability = 0.1f) : probability_(probability) {}
    
    Image apply(const Image& image) override {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(gen_) < probability_) {
            return image.to_grayscale(image.channels());
        }
        return image;
    }
};

/**
 * @brief Gaussian blur
 */
class GaussianBlur : public Transform {
private:
    int kernel_size_;
    float sigma_min_;
    float sigma_max_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    GaussianBlur(int kernel_size, float sigma = 0.0f)
        : kernel_size_(kernel_size), sigma_min_(sigma), sigma_max_(sigma) {
        if (sigma == 0.0f) {
            sigma_min_ = 0.1f;
            sigma_max_ = 2.0f;
        }
    }
    
    GaussianBlur(int kernel_size, float sigma_min, float sigma_max)
        : kernel_size_(kernel_size), sigma_min_(sigma_min), sigma_max_(sigma_max) {}
    
    Image apply(const Image& image) override {
        std::uniform_real_distribution<float> dist(sigma_min_, sigma_max_);
        float sigma = dist(gen_);
        return image.gaussian_blur(kernel_size_, sigma);
    }
};

/**
 * @brief Random erasing (cutout)
 */
class RandomErasing : public Transform {
private:
    float probability_;
    std::pair<float, float> scale_;
    std::pair<float, float> ratio_;
    float value_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandomErasing(float probability = 0.5f,
                  std::pair<float, float> scale = {0.02f, 0.33f},
                  std::pair<float, float> ratio = {0.3f, 3.3f},
                  float value = 0.0f)
        : probability_(probability), scale_(scale), ratio_(ratio), value_(value) {}
    
    Image apply(const Image& image) override {
        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        if (prob_dist(gen_) >= probability_) {
            return image;
        }
        
        int w = image.width();
        int h = image.height();
        float area = static_cast<float>(w * h);
        
        std::uniform_real_distribution<float> scale_dist(scale_.first, scale_.second);
        std::uniform_real_distribution<float> ratio_dist(ratio_.first, ratio_.second);
        
        for (int attempt = 0; attempt < 10; ++attempt) {
            float erase_area = area * scale_dist(gen_);
            float aspect_ratio = ratio_dist(gen_);
            
            int erase_w = static_cast<int>(std::sqrt(erase_area * aspect_ratio));
            int erase_h = static_cast<int>(std::sqrt(erase_area / aspect_ratio));
            
            if (erase_w < w && erase_h < h) {
                std::uniform_int_distribution<int> x_dist(0, w - erase_w);
                std::uniform_int_distribution<int> y_dist(0, h - erase_h);
                
                int x = x_dist(gen_);
                int y = y_dist(gen_);
                
                Image result = image;
                result.fill_rect(x, y, erase_w, erase_h, value_);
                return result;
            }
        }
        
        return image;
    }
};

/**
 * @brief Auto augment policy
 */
class AutoAugment : public Transform {
private:
    std::string policy_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    AutoAugment(const std::string& policy = "imagenet") : policy_(policy) {}
    
    Image apply(const Image& image) override {
        // Simplified auto augment - apply random transforms
        std::uniform_int_distribution<int> transform_dist(0, 5);
        std::uniform_real_distribution<float> magnitude_dist(0.0f, 1.0f);
        
        Image result = image;
        
        for (int i = 0; i < 2; ++i) {
            int t = transform_dist(gen_);
            float m = magnitude_dist(gen_);
            
            switch (t) {
                case 0:
                    result = result.rotate(m * 30.0f - 15.0f);
                    break;
                case 1:
                    result = result.adjust_brightness(0.5f + m);
                    break;
                case 2:
                    result = result.adjust_contrast(0.5f + m);
                    break;
                case 3:
                    if (magnitude_dist(gen_) < 0.5f) {
                        result = result.flip_horizontal();
                    }
                    break;
                case 4:
                    result = result.adjust_saturation(0.5f + m);
                    break;
                case 5:
                    result = result.adjust_sharpness(0.5f + m);
                    break;
            }
        }
        
        return result;
    }
};

/**
 * @brief RandAugment
 */
class RandAugment : public Transform {
private:
    int num_ops_;
    float magnitude_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    RandAugment(int num_ops = 2, float magnitude = 9.0f)
        : num_ops_(num_ops), magnitude_(magnitude / 30.0f) {}
    
    Image apply(const Image& image) override {
        std::uniform_int_distribution<int> op_dist(0, 13);
        
        Image result = image;
        
        for (int i = 0; i < num_ops_; ++i) {
            int op = op_dist(gen_);
            
            switch (op) {
                case 0:  // Identity
                    break;
                case 1:  // AutoContrast
                    result = result.auto_contrast();
                    break;
                case 2:  // Equalize
                    result = result.equalize();
                    break;
                case 3:  // Rotate
                    result = result.rotate(magnitude_ * 30.0f);
                    break;
                case 4:  // Posterize
                    result = result.posterize(static_cast<int>(4 + magnitude_ * 4));
                    break;
                case 5:  // Solarize
                    result = result.solarize(static_cast<int>(256 - magnitude_ * 256));
                    break;
                case 6:  // Color (Saturation)
                    result = result.adjust_saturation(magnitude_);
                    break;
                case 7:  // Contrast
                    result = result.adjust_contrast(magnitude_);
                    break;
                case 8:  // Brightness
                    result = result.adjust_brightness(magnitude_);
                    break;
                case 9:  // Sharpness
                    result = result.adjust_sharpness(magnitude_);
                    break;
                case 10: // ShearX
                    result = result.shear_x(magnitude_ * 0.3f);
                    break;
                case 11: // ShearY
                    result = result.shear_y(magnitude_ * 0.3f);
                    break;
                case 12: // TranslateX
                    result = result.translate(static_cast<int>(magnitude_ * image.width() * 0.3f), 0);
                    break;
                case 13: // TranslateY
                    result = result.translate(0, static_cast<int>(magnitude_ * image.height() * 0.3f));
                    break;
            }
        }
        
        return result;
    }
};

/**
 * @brief Convert image to tensor
 */
class ToTensor : public Transform {
public:
    Image apply(const Image& image) override {
        // Convert to float and normalize to [0, 1]
        Image result = image;
        for (auto& pixel : result.data()) {
            pixel /= 255.0f;
        }
        return result;
    }
};

/**
 * @brief Convert tensor to PIL-style image
 */
class ToPILImage : public Transform {
public:
    Image apply(const Image& image) override {
        // Convert from [0, 1] to [0, 255]
        Image result = image;
        for (auto& pixel : result.data()) {
            pixel = std::max(0.0f, std::min(255.0f, pixel * 255.0f));
        }
        return result;
    }
};

/**
 * @brief Mixup augmentation
 */
class Mixup {
private:
    float alpha_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    Mixup(float alpha = 1.0f) : alpha_(alpha) {}
    
    std::pair<Image, float> apply(const Image& image1, const Image& image2) {
        std::gamma_distribution<float> dist(alpha_, 1.0f);
        float lam = dist(gen_) / (dist(gen_) + dist(gen_));
        
        Image mixed = image1.blend(image2, lam);
        return {mixed, lam};
    }
};

/**
 * @brief CutMix augmentation
 */
class CutMix {
private:
    float alpha_;
    mutable std::mt19937 gen_{std::random_device{}()};
    
public:
    CutMix(float alpha = 1.0f) : alpha_(alpha) {}
    
    std::pair<Image, float> apply(const Image& image1, const Image& image2) {
        std::gamma_distribution<float> dist(alpha_, 1.0f);
        float lam = dist(gen_) / (dist(gen_) + dist(gen_));
        
        int w = image1.width();
        int h = image1.height();
        
        float cut_ratio = std::sqrt(1.0f - lam);
        int cut_w = static_cast<int>(w * cut_ratio);
        int cut_h = static_cast<int>(h * cut_ratio);
        
        std::uniform_int_distribution<int> x_dist(0, w);
        std::uniform_int_distribution<int> y_dist(0, h);
        
        int cx = x_dist(gen_);
        int cy = y_dist(gen_);
        
        int x1 = std::max(0, cx - cut_w / 2);
        int y1 = std::max(0, cy - cut_h / 2);
        int x2 = std::min(w, cx + cut_w / 2);
        int y2 = std::min(h, cy + cut_h / 2);
        
        Image result = image1;
        result.paste(image2, x1, y1, x2 - x1, y2 - y1);
        
        float actual_lam = 1.0f - static_cast<float>((x2 - x1) * (y2 - y1)) / (w * h);
        
        return {result, actual_lam};
    }
};

} // namespace augmentation
} // namespace neurova
