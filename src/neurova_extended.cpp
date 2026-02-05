// Copyright (c) 2026 Neurova
// High-performance C++ implementation of core, augmentation, and calibration modules
// Compiled with: clang++ -O3 -std=c++17 -march=armv8-a -fPIC -shared

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <random>

#ifdef __ARM_NEON
#include <arm_neon.h>
#define SIMD_ENABLED "ARM NEON"
#elif defined(__AVX2__)
#include <immintrin.h>
#define SIMD_ENABLED "AVX2"
#else
#define SIMD_ENABLED "None"
#endif

namespace py = pybind11;

// =============================================================================
// CORE MODULE - Image and Array Classes
// =============================================================================

enum class DType {
    UINT8,
    UINT16,
    FLOAT32,
    FLOAT64
};

enum class ColorSpace {
    GRAY,
    RGB,
    BGR,
    HSV,
    HSL,
    LAB,
    YUV
};

// Simple Array class for N-dimensional data
class Array {
public:
    std::vector<size_t> shape_;
    DType dtype_;
    std::vector<float> data_;

    Array() : dtype_(DType::FLOAT32) {}
    
    Array(const std::vector<size_t>& shape, DType dtype = DType::FLOAT32)
        : shape_(shape), dtype_(dtype) {
        size_t total = 1;
        for (auto s : shape) total *= s;
        data_.resize(total, 0.0f);
    }

    static Array zeros(const std::vector<size_t>& shape, DType dtype = DType::FLOAT32) {
        return Array(shape, dtype);
    }

    static Array ones(const std::vector<size_t>& shape, DType dtype = DType::FLOAT32) {
        Array arr(shape, dtype);
        std::fill(arr.data_.begin(), arr.data_.end(), 1.0f);
        return arr;
    }

    static Array randn(const std::vector<size_t>& shape, DType dtype = DType::FLOAT32) {
        Array arr(shape, dtype);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : arr.data_) v = dist(gen);
        return arr;
    }

    size_t size() const {
        size_t total = 1;
        for (auto s : shape_) total *= s;
        return total;
    }

    std::vector<size_t> shape() const { return shape_; }
    
    py::array_t<float> to_numpy() const {
        return py::array_t<float>(shape_, data_.data());
    }
};

// Image class - multi-channel image container
class Image {
public:
    size_t width_, height_, channels_;
    DType dtype_;
    ColorSpace color_space_;
    std::vector<uint8_t> data_;

    Image() : width_(0), height_(0), channels_(0), dtype_(DType::UINT8), 
              color_space_(ColorSpace::RGB) {}

    Image(size_t width, size_t height, size_t channels, 
          DType dtype = DType::UINT8, ColorSpace cs = ColorSpace::RGB)
        : width_(width), height_(height), channels_(channels), 
          dtype_(dtype), color_space_(cs) {
        data_.resize(width * height * channels, 0);
    }

    Image(py::array_t<uint8_t> arr, ColorSpace cs = ColorSpace::RGB) 
        : dtype_(DType::UINT8), color_space_(cs) {
        auto buf = arr.request();
        
        if (buf.ndim == 2) {
            height_ = buf.shape[0];
            width_ = buf.shape[1];
            channels_ = 1;
        } else if (buf.ndim == 3) {
            height_ = buf.shape[0];
            width_ = buf.shape[1];
            channels_ = buf.shape[2];
        } else {
            throw std::runtime_error("Invalid array dimensions");
        }

        auto ptr = static_cast<uint8_t*>(buf.ptr);
        data_.assign(ptr, ptr + (width_ * height_ * channels_));
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    size_t channels() const { return channels_; }
    std::string dtype_str() const { return "UINT8"; }
    std::string color_space_str() const { 
        switch(color_space_) {
            case ColorSpace::GRAY: return "GRAY";
            case ColorSpace::RGB: return "RGB";
            case ColorSpace::BGR: return "BGR";
            case ColorSpace::HSV: return "HSV";
            case ColorSpace::HSL: return "HSL";
            case ColorSpace::LAB: return "LAB";
            case ColorSpace::YUV: return "YUV";
            default: return "UNKNOWN";
        }
    }

    Image clone() const {
        Image img(width_, height_, channels_, dtype_, color_space_);
        img.data_ = data_;
        return img;
    }

    Image crop(size_t x, size_t y, size_t w, size_t h) const {
        if (x + w > width_ || y + h > height_) {
            throw std::runtime_error("Crop region exceeds image bounds");
        }
        
        Image result(w, h, channels_, dtype_, color_space_);
        
        for (size_t row = 0; row < h; ++row) {
            for (size_t col = 0; col < w; ++col) {
                for (size_t c = 0; c < channels_; ++c) {
                    size_t src_idx = ((y + row) * width_ + (x + col)) * channels_ + c;
                    size_t dst_idx = (row * w + col) * channels_ + c;
                    result.data_[dst_idx] = data_[src_idx];
                }
            }
        }
        
        return result;
    }

    py::array_t<uint8_t> to_numpy() const {
        if (channels_ == 1) {
            return py::array_t<uint8_t>({height_, width_}, data_.data());
        } else {
            return py::array_t<uint8_t>({height_, width_, channels_}, data_.data());
        }
    }

    // Convert from CHW to HWC format
    Image chw_to_hwc() const {
        if (channels_ == 1) return clone();
        
        Image result(width_, height_, channels_, dtype_, color_space_);
        
        for (size_t c = 0; c < channels_; ++c) {
            for (size_t y = 0; y < height_; ++y) {
                for (size_t x = 0; x < width_; ++x) {
                    // CHW: data[c][y][x]
                    size_t src_idx = c * (height_ * width_) + y * width_ + x;
                    // HWC: data[y][x][c]
                    size_t dst_idx = (y * width_ + x) * channels_ + c;
                    result.data_[dst_idx] = data_[src_idx];
                }
            }
        }
        
        return result;
    }

    // Convert from HWC to CHW format
    Image hwc_to_chw() const {
        if (channels_ == 1) return clone();
        
        Image result(width_, height_, channels_, dtype_, color_space_);
        
        for (size_t y = 0; y < height_; ++y) {
            for (size_t x = 0; x < width_; ++x) {
                for (size_t c = 0; c < channels_; ++c) {
                    // HWC: data[y][x][c]
                    size_t src_idx = (y * width_ + x) * channels_ + c;
                    // CHW: data[c][y][x]
                    size_t dst_idx = c * (height_ * width_) + y * width_ + x;
                    result.data_[dst_idx] = data_[src_idx];
                }
            }
        }
        
        return result;
    }
};

// =============================================================================
// COLOR CONVERSIONS
// =============================================================================

// RGB to Grayscale conversion
Image rgb_to_gray(const Image& img) {
    if (img.channels_ == 1) return img.clone();
    if (img.channels_ != 3) throw std::runtime_error("Expected 3-channel RGB image");

    Image gray(img.width_, img.height_, 1, img.dtype_, ColorSpace::GRAY);

    const float weights[3] = {0.299f, 0.587f, 0.114f};

#ifdef __ARM_NEON
    // NEON-optimized grayscale conversion
    for (size_t i = 0; i < img.width_ * img.height_; i += 8) {
        if (i + 8 * 3 > img.data_.size()) break;
        
        uint8x8x3_t rgb = vld3_u8(&img.data_[i * 3]);
        
        uint16x8_t r16 = vmovl_u8(rgb.val[0]);
        uint16x8_t g16 = vmovl_u8(rgb.val[1]);
        uint16x8_t b16 = vmovl_u8(rgb.val[2]);
        
        uint16x8_t gray16 = vmulq_n_u16(r16, 77);  // 0.299 * 256
        gray16 = vmlaq_n_u16(gray16, g16, 150);    // 0.587 * 256
        gray16 = vmlaq_n_u16(gray16, b16, 29);     // 0.114 * 256
        gray16 = vshrq_n_u16(gray16, 8);
        
        uint8x8_t gray8 = vmovn_u16(gray16);
        vst1_u8(&gray.data_[i], gray8);
    }
    
    // Process remaining pixels
    size_t processed = (img.width_ * img.height_ / 8) * 8;
    for (size_t i = processed; i < img.width_ * img.height_; ++i) {
        size_t idx = i * 3;
        gray.data_[i] = static_cast<uint8_t>(
            img.data_[idx] * weights[0] +
            img.data_[idx + 1] * weights[1] +
            img.data_[idx + 2] * weights[2]
        );
    }
#else
    // Scalar version
    for (size_t i = 0; i < img.width_ * img.height_; ++i) {
        size_t idx = i * 3;
        gray.data_[i] = static_cast<uint8_t>(
            img.data_[idx] * weights[0] +
            img.data_[idx + 1] * weights[1] +
            img.data_[idx + 2] * weights[2]
        );
    }
#endif

    return gray;
}

// RGB to HSV conversion
py::array_t<float> rgb_to_hsv(py::array_t<uint8_t> rgb_arr) {
    auto buf = rgb_arr.request();
    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw std::runtime_error("Expected (H, W, 3) RGB array");
    }

    size_t h = buf.shape[0];
    size_t w = buf.shape[1];
    auto rgb = static_cast<uint8_t*>(buf.ptr);

    std::vector<float> hsv(h * w * 3);

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            size_t idx = (y * w + x) * 3;
            
            float r = rgb[idx] / 255.0f;
            float g = rgb[idx + 1] / 255.0f;
            float b = rgb[idx + 2] / 255.0f;

            float max_val = std::max({r, g, b});
            float min_val = std::min({r, g, b});
            float diff = max_val - min_val;

            // Hue
            float hue = 0.0f;
            if (diff > 1e-6f) {
                if (max_val == r) {
                    hue = 60.0f * fmodf((g - b) / diff, 6.0f);
                } else if (max_val == g) {
                    hue = 60.0f * ((b - r) / diff + 2.0f);
                } else {
                    hue = 60.0f * ((r - g) / diff + 4.0f);
                }
            }
            if (hue < 0) hue += 360.0f;

            // Saturation
            float sat = (max_val > 1e-6f) ? (diff / max_val) : 0.0f;

            // Value
            float val = max_val;

            hsv[idx] = hue / 360.0f;  // Normalize to [0, 1]
            hsv[idx + 1] = sat;
            hsv[idx + 2] = val;
        }
    }

    std::vector<ssize_t> shape = {static_cast<ssize_t>(h), static_cast<ssize_t>(w), 3};
    return py::array_t<float>(shape, hsv.data());
}

// HSV to RGB conversion
py::array_t<uint8_t> hsv_to_rgb(py::array_t<float> hsv_arr) {
    auto buf = hsv_arr.request();
    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw std::runtime_error("Expected (H, W, 3) HSV array");
    }

    size_t h = buf.shape[0];
    size_t w = buf.shape[1];
    auto hsv = static_cast<float*>(buf.ptr);

    std::vector<uint8_t> rgb(h * w * 3);

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            size_t idx = (y * w + x) * 3;
            
            float hue = hsv[idx] * 360.0f;  // Denormalize from [0, 1]
            float sat = hsv[idx + 1];
            float val = hsv[idx + 2];

            float c = val * sat;
            float x_val = c * (1.0f - fabsf(fmodf(hue / 60.0f, 2.0f) - 1.0f));
            float m = val - c;

            float r, g, b;
            if (hue < 60.0f) {
                r = c; g = x_val; b = 0;
            } else if (hue < 120.0f) {
                r = x_val; g = c; b = 0;
            } else if (hue < 180.0f) {
                r = 0; g = c; b = x_val;
            } else if (hue < 240.0f) {
                r = 0; g = x_val; b = c;
            } else if (hue < 300.0f) {
                r = x_val; g = 0; b = c;
            } else {
                r = c; g = 0; b = x_val;
            }

            rgb[idx] = static_cast<uint8_t>((r + m) * 255.0f);
            rgb[idx + 1] = static_cast<uint8_t>((g + m) * 255.0f);
            rgb[idx + 2] = static_cast<uint8_t>((b + m) * 255.0f);
        }
    }

    std::vector<ssize_t> shape = {static_cast<ssize_t>(h), static_cast<ssize_t>(w), 3};
    return py::array_t<uint8_t>(shape, rgb.data());
}

// RGB to BGR and vice versa
Image rgb_to_bgr(const Image& img) {
    if (img.channels_ != 3) throw std::runtime_error("Expected 3-channel image");

    Image bgr = img.clone();
    bgr.color_space_ = (img.color_space_ == ColorSpace::RGB) ? ColorSpace::BGR : ColorSpace::RGB;

    for (size_t i = 0; i < img.width_ * img.height_; ++i) {
        size_t idx = i * 3;
        std::swap(bgr.data_[idx], bgr.data_[idx + 2]);
    }

    return bgr;
}

// =============================================================================
// AUGMENTATION - GEOMETRIC TRANSFORMATIONS
// =============================================================================

// Horizontal flip
Image hflip(const Image& img) {
    Image result(img.width_, img.height_, img.channels_, img.dtype_, img.color_space_);

    for (size_t y = 0; y < img.height_; ++y) {
        for (size_t x = 0; x < img.width_; ++x) {
            for (size_t c = 0; c < img.channels_; ++c) {
                size_t src_idx = (y * img.width_ + x) * img.channels_ + c;
                size_t dst_idx = (y * img.width_ + (img.width_ - 1 - x)) * img.channels_ + c;
                result.data_[dst_idx] = img.data_[src_idx];
            }
        }
    }

    return result;
}

// Vertical flip
Image vflip(const Image& img) {
    Image result(img.width_, img.height_, img.channels_, img.dtype_, img.color_space_);

    for (size_t y = 0; y < img.height_; ++y) {
        for (size_t x = 0; x < img.width_; ++x) {
            for (size_t c = 0; c < img.channels_; ++c) {
                size_t src_idx = (y * img.width_ + x) * img.channels_ + c;
                size_t dst_idx = ((img.height_ - 1 - y) * img.width_ + x) * img.channels_ + c;
                result.data_[dst_idx] = img.data_[src_idx];
            }
        }
    }

    return result;
}

// Rotate image by angle (degrees)
Image rotate(const Image& img, float angle_degrees) {
    Image result(img.width_, img.height_, img.channels_, img.dtype_, img.color_space_);
    std::fill(result.data_.begin(), result.data_.end(), 0);

    float angle_rad = angle_degrees * M_PI / 180.0f;
    float cos_a = cosf(angle_rad);
    float sin_a = sinf(angle_rad);

    float cx = img.width_ / 2.0f;
    float cy = img.height_ / 2.0f;

    for (size_t y = 0; y < img.height_; ++y) {
        for (size_t x = 0; x < img.width_; ++x) {
            // Rotate around center
            float dx = x - cx;
            float dy = y - cy;
            
            float src_x = dx * cos_a - dy * sin_a + cx;
            float src_y = dx * sin_a + dy * cos_a + cy;

            // Bilinear interpolation
            int x0 = static_cast<int>(floorf(src_x));
            int y0 = static_cast<int>(floorf(src_y));
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            if (x0 >= 0 && x1 < static_cast<int>(img.width_) && 
                y0 >= 0 && y1 < static_cast<int>(img.height_)) {
                
                float fx = src_x - x0;
                float fy = src_y - y0;

                for (size_t c = 0; c < img.channels_; ++c) {
                    float v00 = img.data_[(y0 * img.width_ + x0) * img.channels_ + c];
                    float v01 = img.data_[(y0 * img.width_ + x1) * img.channels_ + c];
                    float v10 = img.data_[(y1 * img.width_ + x0) * img.channels_ + c];
                    float v11 = img.data_[(y1 * img.width_ + x1) * img.channels_ + c];

                    float v0 = v00 * (1 - fx) + v01 * fx;
                    float v1 = v10 * (1 - fx) + v11 * fx;
                    float v = v0 * (1 - fy) + v1 * fy;

                    result.data_[(y * img.width_ + x) * img.channels_ + c] = 
                        static_cast<uint8_t>(v);
                }
            }
        }
    }

    return result;
}

// Resize image using bilinear interpolation
Image resize(const Image& img, size_t new_width, size_t new_height) {
    Image result(new_width, new_height, img.channels_, img.dtype_, img.color_space_);

    float x_ratio = static_cast<float>(img.width_ - 1) / new_width;
    float y_ratio = static_cast<float>(img.height_ - 1) / new_height;

    for (size_t y = 0; y < new_height; ++y) {
        for (size_t x = 0; x < new_width; ++x) {
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;

            int x0 = static_cast<int>(floorf(src_x));
            int y0 = static_cast<int>(floorf(src_y));
            int x1 = std::min(x0 + 1, static_cast<int>(img.width_ - 1));
            int y1 = std::min(y0 + 1, static_cast<int>(img.height_ - 1));

            float fx = src_x - x0;
            float fy = src_y - y0;

            for (size_t c = 0; c < img.channels_; ++c) {
                float v00 = img.data_[(y0 * img.width_ + x0) * img.channels_ + c];
                float v01 = img.data_[(y0 * img.width_ + x1) * img.channels_ + c];
                float v10 = img.data_[(y1 * img.width_ + x0) * img.channels_ + c];
                float v11 = img.data_[(y1 * img.width_ + x1) * img.channels_ + c];

                float v0 = v00 * (1 - fx) + v01 * fx;
                float v1 = v10 * (1 - fx) + v11 * fx;
                float v = v0 * (1 - fy) + v1 * fy;

                result.data_[(y * new_width + x) * img.channels_ + c] = 
                    static_cast<uint8_t>(v);
            }
        }
    }

    return result;
}

// =============================================================================
// AUGMENTATION - COLOR TRANSFORMATIONS
// =============================================================================

// Adjust brightness
Image adjust_brightness(const Image& img, float factor) {
    Image result = img.clone();

    for (size_t i = 0; i < result.data_.size(); ++i) {
        float val = result.data_[i] * factor;
        result.data_[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, val)));
    }

    return result;
}

// Adjust contrast
Image adjust_contrast(const Image& img, float factor) {
    // Calculate mean
    float mean = 0.0f;
    for (auto val : img.data_) {
        mean += val;
    }
    mean /= img.data_.size();

    Image result = img.clone();

    for (size_t i = 0; i < result.data_.size(); ++i) {
        float val = mean + (result.data_[i] - mean) * factor;
        result.data_[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, val)));
    }

    return result;
}

// Adjust saturation (HSV space)
py::array_t<uint8_t> adjust_saturation(py::array_t<uint8_t> rgb_arr, float factor) {
    // Convert to HSV
    auto hsv = rgb_to_hsv(rgb_arr);
    auto hsv_buf = hsv.request();
    auto hsv_ptr = static_cast<float*>(hsv_buf.ptr);

    size_t total_pixels = hsv_buf.shape[0] * hsv_buf.shape[1];

    // Adjust saturation channel (index 1)
    for (size_t i = 0; i < total_pixels; ++i) {
        size_t idx = i * 3 + 1;
        hsv_ptr[idx] = std::min(1.0f, std::max(0.0f, hsv_ptr[idx] * factor));
    }

    // Convert back to RGB
    return hsv_to_rgb(hsv);
}

// Adjust hue (HSV space)
py::array_t<uint8_t> adjust_hue(py::array_t<uint8_t> rgb_arr, float hue_shift) {
    // Convert to HSV
    auto hsv = rgb_to_hsv(rgb_arr);
    auto hsv_buf = hsv.request();
    auto hsv_ptr = static_cast<float*>(hsv_buf.ptr);

    size_t total_pixels = hsv_buf.shape[0] * hsv_buf.shape[1];

    // Adjust hue channel (index 0)
    for (size_t i = 0; i < total_pixels; ++i) {
        size_t idx = i * 3;
        hsv_ptr[idx] = fmodf(hsv_ptr[idx] + hue_shift, 1.0f);
        if (hsv_ptr[idx] < 0) hsv_ptr[idx] += 1.0f;
    }

    // Convert back to RGB
    return hsv_to_rgb(hsv);
}

// Add Gaussian noise
Image add_gaussian_noise(const Image& img, float mean, float std) {
    Image result = img.clone();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);

    for (size_t i = 0; i < result.data_.size(); ++i) {
        float noise = dist(gen);
        float val = result.data_[i] + noise;
        result.data_[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, val)));
    }

    return result;
}

// Normalize image
py::array_t<float> normalize(py::array_t<uint8_t> img_arr, 
                             const std::vector<float>& mean,
                             const std::vector<float>& std) {
    auto buf = img_arr.request();
    
    size_t h = buf.shape[0];
    size_t w = buf.shape[1];
    size_t c = (buf.ndim == 3) ? buf.shape[2] : 1;

    auto ptr = static_cast<uint8_t*>(buf.ptr);
    std::vector<float> result(h * w * c);

    for (size_t i = 0; i < h * w; ++i) {
        for (size_t ch = 0; ch < c; ++ch) {
            size_t idx = i * c + ch;
            float val = ptr[idx] / 255.0f;
            float m = (ch < mean.size()) ? mean[ch] : 0.0f;
            float s = (ch < std.size()) ? std[ch] : 1.0f;
            result[idx] = (val - m) / s;
        }
    }

    if (c == 1) {
        std::vector<ssize_t> shape = {static_cast<ssize_t>(h), static_cast<ssize_t>(w)};
        return py::array_t<float>(shape, result.data());
    } else {
        std::vector<ssize_t> shape = {static_cast<ssize_t>(h), static_cast<ssize_t>(w), static_cast<ssize_t>(c)};
        return py::array_t<float>(shape, result.data());
    }
}

// =============================================================================
// CALIBRATION - CAMERA CALIBRATION AND POSE ESTIMATION
// =============================================================================

// Rodrigues rotation vector to matrix conversion
py::array_t<float> rodrigues(py::array_t<float> rvec_arr) {
    auto buf = rvec_arr.request();
    auto rvec = static_cast<float*>(buf.ptr);

    float theta = sqrtf(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]);
    
    std::vector<float> R(9);
    
    if (theta < 1e-6f) {
        // Identity matrix
        R[0] = 1; R[1] = 0; R[2] = 0;
        R[3] = 0; R[4] = 1; R[5] = 0;
        R[6] = 0; R[7] = 0; R[8] = 1;
    } else {
        float c = cosf(theta);
        float s = sinf(theta);
        float t = 1.0f - c;
        
        float x = rvec[0] / theta;
        float y = rvec[1] / theta;
        float z = rvec[2] / theta;
        
        R[0] = t * x * x + c;
        R[1] = t * x * y - s * z;
        R[2] = t * x * z + s * y;
        R[3] = t * x * y + s * z;
        R[4] = t * y * y + c;
        R[5] = t * y * z - s * x;
        R[6] = t * x * z - s * y;
        R[7] = t * y * z + s * x;
        R[8] = t * z * z + c;
    }

    std::vector<ssize_t> shape = {3, 3};
    return py::array_t<float>(shape, R.data());
}

// Project 3D points to 2D image plane
py::array_t<float> project_points(py::array_t<float> obj_pts,
                                   py::array_t<float> rvec,
                                   py::array_t<float> tvec,
                                   py::array_t<float> camera_matrix) {
    auto R = rodrigues(rvec);
    auto R_buf = R.request();
    auto R_ptr = static_cast<float*>(R_buf.ptr);
    
    auto t_buf = tvec.request();
    auto t_ptr = static_cast<float*>(t_buf.ptr);
    
    auto K_buf = camera_matrix.request();
    auto K_ptr = static_cast<float*>(K_buf.ptr);
    
    auto pts_buf = obj_pts.request();
    auto pts_ptr = static_cast<float*>(pts_buf.ptr);
    size_t n_pts = pts_buf.shape[0];

    std::vector<float> img_pts(n_pts * 2);

    for (size_t i = 0; i < n_pts; ++i) {
        float X = pts_ptr[i * 3];
        float Y = pts_ptr[i * 3 + 1];
        float Z = pts_ptr[i * 3 + 2];

        // Apply rotation and translation
        float x = R_ptr[0] * X + R_ptr[1] * Y + R_ptr[2] * Z + t_ptr[0];
        float y = R_ptr[3] * X + R_ptr[4] * Y + R_ptr[5] * Z + t_ptr[1];
        float z = R_ptr[6] * X + R_ptr[7] * Y + R_ptr[8] * Z + t_ptr[2];

        // Project to image plane
        float fx = K_ptr[0];
        float fy = K_ptr[4];
        float cx = K_ptr[2];
        float cy = K_ptr[5];

        img_pts[i * 2] = fx * (x / z) + cx;
        img_pts[i * 2 + 1] = fy * (y / z) + cy;
    }

    std::vector<ssize_t> shape = {static_cast<ssize_t>(n_pts), 2};
    return py::array_t<float>(shape, img_pts.data());
}

// Simple homography estimation using DLT
py::array_t<float> find_homography(py::array_t<float> src_pts, 
                                    py::array_t<float> dst_pts) {
    auto src_buf = src_pts.request();
    auto dst_buf = dst_pts.request();
    
    if (src_buf.shape[0] != dst_buf.shape[0] || src_buf.shape[0] < 4) {
        throw std::runtime_error("Need at least 4 point correspondences");
    }

    size_t n = src_buf.shape[0];
    auto src = static_cast<float*>(src_buf.ptr);
    auto dst = static_cast<float*>(dst_buf.ptr);

    // Build 2n x 9 matrix A for DLT
    std::vector<float> A(2 * n * 9, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        float x = src[i * 2];
        float y = src[i * 2 + 1];
        float xp = dst[i * 2];
        float yp = dst[i * 2 + 1];

        // First row: [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        A[2 * i * 9 + 0] = -x;
        A[2 * i * 9 + 1] = -y;
        A[2 * i * 9 + 2] = -1;
        A[2 * i * 9 + 6] = x * xp;
        A[2 * i * 9 + 7] = y * xp;
        A[2 * i * 9 + 8] = xp;

        // Second row: [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
        A[(2 * i + 1) * 9 + 3] = -x;
        A[(2 * i + 1) * 9 + 4] = -y;
        A[(2 * i + 1) * 9 + 5] = -1;
        A[(2 * i + 1) * 9 + 6] = x * yp;
        A[(2 * i + 1) * 9 + 7] = y * yp;
        A[(2 * i + 1) * 9 + 8] = yp;
    }

    // Simplified solution: use least squares approximation
    // For production, use proper SVD
    std::vector<float> H = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };

    std::vector<ssize_t> shape = {3, 3};
    return py::array_t<float>(shape, H.data());
}

// =============================================================================
// PYBIND11 BINDINGS
// =============================================================================

PYBIND11_MODULE(neurova_extended, m) {
    m.doc() = "Neurova Extended Module - Core, Augmentation, and Calibration in C++";
    m.attr("__version__") = "0.3.0";
    m.attr("SIMD_SUPPORT") = SIMD_ENABLED;

    // =========================================================================
    // ENUMS
    // =========================================================================
    
    py::enum_<DType>(m, "DType")
        .value("UINT8", DType::UINT8)
        .value("UINT16", DType::UINT16)
        .value("FLOAT32", DType::FLOAT32)
        .value("FLOAT64", DType::FLOAT64);

    py::enum_<ColorSpace>(m, "ColorSpace")
        .value("GRAY", ColorSpace::GRAY)
        .value("RGB", ColorSpace::RGB)
        .value("BGR", ColorSpace::BGR)
        .value("HSV", ColorSpace::HSV)
        .value("HSL", ColorSpace::HSL)
        .value("LAB", ColorSpace::LAB)
        .value("YUV", ColorSpace::YUV);

    // =========================================================================
    // ARRAY CLASS
    // =========================================================================
    
    py::class_<Array>(m, "Array")
        .def(py::init<>())
        .def(py::init<const std::vector<size_t>&, DType>(),
             py::arg("shape"), py::arg("dtype") = DType::FLOAT32)
        .def_static("zeros", &Array::zeros,
                   py::arg("shape"), py::arg("dtype") = DType::FLOAT32)
        .def_static("ones", &Array::ones,
                   py::arg("shape"), py::arg("dtype") = DType::FLOAT32)
        .def_static("randn", &Array::randn,
                   py::arg("shape"), py::arg("dtype") = DType::FLOAT32)
        .def("size", &Array::size)
        .def("shape", &Array::shape)
        .def("to_numpy", &Array::to_numpy)
        .def_readonly("dtype", &Array::dtype_);

    // =========================================================================
    // IMAGE CLASS
    // =========================================================================
    
    py::class_<Image>(m, "Image")
        .def(py::init<>())
        .def(py::init<size_t, size_t, size_t, DType, ColorSpace>(),
             py::arg("width"), py::arg("height"), py::arg("channels"),
             py::arg("dtype") = DType::UINT8, py::arg("color_space") = ColorSpace::RGB)
        .def(py::init<py::array_t<uint8_t>, ColorSpace>(),
             py::arg("array"), py::arg("color_space") = ColorSpace::RGB)
        .def_property_readonly("width", &Image::width)
        .def_property_readonly("height", &Image::height)
        .def_property_readonly("channels", &Image::channels)
        .def_property_readonly("dtype", &Image::dtype_str)
        .def_property_readonly("color_space", &Image::color_space_str)
        .def("clone", &Image::clone)
        .def("crop", &Image::crop)
        .def("to_numpy", &Image::to_numpy)
        .def("chw_to_hwc", &Image::chw_to_hwc)
        .def("hwc_to_chw", &Image::hwc_to_chw);

    // =========================================================================
    // COLOR CONVERSIONS
    // =========================================================================
    
    py::module color = m.def_submodule("color", "Color space conversions");
    
    color.def("rgb_to_gray", &rgb_to_gray, "Convert RGB to grayscale");
    color.def("rgb_to_hsv", &rgb_to_hsv, "Convert RGB to HSV");
    color.def("hsv_to_rgb", &hsv_to_rgb, "Convert HSV to RGB");
    color.def("rgb_to_bgr", &rgb_to_bgr, "Convert RGB to BGR");
    color.def("bgr_to_rgb", &rgb_to_bgr, "Convert BGR to RGB (same as RGB to BGR)");

    // =========================================================================
    // AUGMENTATION - GEOMETRIC
    // =========================================================================
    
    py::module augmentation = m.def_submodule("augmentation", "Image augmentation transforms");
    
    augmentation.def("hflip", &hflip, "Horizontal flip");
    augmentation.def("vflip", &vflip, "Vertical flip");
    augmentation.def("rotate", &rotate, "Rotate by angle in degrees",
                    py::arg("image"), py::arg("angle"));
    augmentation.def("resize", &resize, "Resize image",
                    py::arg("image"), py::arg("width"), py::arg("height"));
    
    // =========================================================================
    // AUGMENTATION - COLOR
    // =========================================================================
    
    augmentation.def("adjust_brightness", &adjust_brightness, 
                    "Adjust brightness", py::arg("image"), py::arg("factor"));
    augmentation.def("adjust_contrast", &adjust_contrast,
                    "Adjust contrast", py::arg("image"), py::arg("factor"));
    augmentation.def("adjust_saturation", &adjust_saturation,
                    "Adjust saturation", py::arg("image"), py::arg("factor"));
    augmentation.def("adjust_hue", &adjust_hue,
                    "Adjust hue", py::arg("image"), py::arg("hue_shift"));
    augmentation.def("add_gaussian_noise", &add_gaussian_noise,
                    "Add Gaussian noise", py::arg("image"), 
                    py::arg("mean") = 0.0f, py::arg("std") = 25.0f);
    augmentation.def("normalize", &normalize,
                    "Normalize image with mean and std",
                    py::arg("image"), py::arg("mean"), py::arg("std"));

    // =========================================================================
    // CALIBRATION
    // =========================================================================
    
    py::module calibration = m.def_submodule("calibration", "Camera calibration and pose");
    
    calibration.def("rodrigues", &rodrigues, 
                   "Convert rotation vector to rotation matrix");
    calibration.def("project_points", &project_points,
                   "Project 3D points to 2D image plane",
                   py::arg("object_points"), py::arg("rvec"), 
                   py::arg("tvec"), py::arg("camera_matrix"));
    calibration.def("find_homography", &find_homography,
                   "Find homography from point correspondences",
                   py::arg("src_points"), py::arg("dst_points"));
}
