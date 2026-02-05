// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file face_detector.hpp
 * @brief Face detection with multiple backend support
 */

#ifndef NEUROVA_FACE_DETECTOR_HPP
#define NEUROVA_FACE_DETECTOR_HPP

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>

#include "../detection/haar_cascade.hpp"
#include "../detection/hog.hpp"

namespace neurova {
namespace face {

/**
 * @brief Face detection result
 */
struct FaceRect {
    int x, y, width, height;
    float confidence;
    
    FaceRect(int x_ = 0, int y_ = 0, int w_ = 0, int h_ = 0, float conf_ = 1.0f)
        : x(x_), y(y_), width(w_), height(h_), confidence(conf_) {}
};

/**
 * @brief Abstract base class for face detectors
 */
class FaceDetectorBase {
public:
    virtual ~FaceDetectorBase() = default;
    
    virtual std::vector<FaceRect> detect(
        const float* gray_image, int width, int height,
        int min_size = 30, int max_size = 0
    ) = 0;
};

/**
 * @brief Haar Cascade face detector
 */
class HaarCascadeDetector : public FaceDetectorBase {
public:
    /**
     * @brief Construct with cascade file path
     */
    explicit HaarCascadeDetector(const std::string& cascade_path)
        : cascade_(cascade_path) {}
    
    std::vector<FaceRect> detect(
        const float* gray_image, int width, int height,
        int min_size = 30, int max_size = 0
    ) override {
        std::vector<FaceRect> faces;
        
        if (cascade_.empty()) return faces;
        
        std::vector<std::tuple<int, int, int, int>> detections;
        cascade_.detectMultiScale(
            gray_image, width, height,
            detections,
            1.1f,           // scale factor
            3,              // min neighbors
            min_size, min_size,
            max_size, max_size
        );
        
        for (const auto& [x, y, w, h] : detections) {
            faces.emplace_back(x, y, w, h, 1.0f);
        }
        
        return faces;
    }
    
private:
    detection::HaarCascadeClassifier cascade_;
};

/**
 * @brief HOG-based face detector
 */
class HOGDetector : public FaceDetectorBase {
public:
    HOGDetector() {
        // Initialize with default parameters for face detection
        hog_ = std::make_unique<detection::HOGDescriptor>(
            64, 128,    // window size
            16, 16,     // block size
            8, 8,       // block stride
            8, 8,       // cell size
            9           // nbins
        );
    }
    
    std::vector<FaceRect> detect(
        const float* gray_image, int width, int height,
        int min_size = 30, int max_size = 0
    ) override {
        std::vector<FaceRect> faces;
        
        // HOG detector would need trained SVM weights
        // This is a placeholder implementation
        
        std::vector<std::tuple<int, int, int, int>> detections;
        std::vector<float> weights;
        
        hog_->detectMultiScale(
            gray_image, width, height,
            detections, weights,
            0.0f,   // hit threshold
            1.05f,  // scale
            2       // group threshold
        );
        
        for (size_t i = 0; i < detections.size(); ++i) {
            auto [x, y, w, h] = detections[i];
            float conf = i < weights.size() ? weights[i] : 1.0f;
            faces.emplace_back(x, y, w, h, conf);
        }
        
        return faces;
    }
    
private:
    std::unique_ptr<detection::HOGDescriptor> hog_;
};

/**
 * @brief Unified face detector with multiple backend support
 */
class FaceDetector {
public:
    enum class Method {
        HAAR,
        LBP,
        HOG,
        DNN,
        NATIVE
    };
    
    /**
     * @brief Construct face detector
     * @param method Detection method
     * @param cascade_path Path to cascade file (for HAAR/LBP)
     * @param min_confidence Minimum detection confidence
     */
    FaceDetector(
        Method method = Method::HAAR,
        const std::string& cascade_path = "",
        float min_confidence = 0.5f
    ) : method_(method), min_confidence_(min_confidence) {
        
        switch (method) {
            case Method::HAAR:
                if (!cascade_path.empty()) {
                    detector_ = std::make_unique<HaarCascadeDetector>(cascade_path);
                }
                break;
            case Method::HOG:
                detector_ = std::make_unique<HOGDetector>();
                break;
            default:
                // Fallback to HOG
                detector_ = std::make_unique<HOGDetector>();
                break;
        }
    }
    
    /**
     * @brief Detect faces in grayscale image
     * @param gray_image Grayscale image (row-major)
     * @param width Image width
     * @param height Image height
     * @param min_size Minimum face size
     * @param max_size Maximum face size (0 = no limit)
     * @return Vector of face rectangles
     */
    std::vector<FaceRect> detect(
        const float* gray_image, int width, int height,
        int min_size = 30, int max_size = 0
    ) {
        if (!detector_) return {};
        
        auto faces = detector_->detect(gray_image, width, height, min_size, max_size);
        
        // Filter by confidence
        faces.erase(
            std::remove_if(faces.begin(), faces.end(),
                [this](const FaceRect& f) { return f.confidence < min_confidence_; }),
            faces.end()
        );
        
        return faces;
    }
    
    /**
     * @brief Detect and crop faces from image
     * @param image Image data (HWC format)
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param margin Margin around face (fraction)
     * @param output_size Output face size (0 = no resize)
     * @return Vector of cropped face images (flattened)
     */
    std::vector<std::vector<float>> detectAndCrop(
        const float* image, int width, int height, int channels,
        float margin = 0.2f, int output_size = 0
    ) {
        // Convert to grayscale
        std::vector<float> gray(width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (channels >= 3) {
                    gray[y * width + x] = 0.299f * image[(y * width + x) * channels + 0] +
                                          0.587f * image[(y * width + x) * channels + 1] +
                                          0.114f * image[(y * width + x) * channels + 2];
                } else {
                    gray[y * width + x] = image[(y * width + x) * channels];
                }
            }
        }
        
        auto faces = detect(gray.data(), width, height);
        
        std::vector<std::vector<float>> cropped;
        
        for (const auto& face : faces) {
            int mx = static_cast<int>(face.width * margin);
            int my = static_cast<int>(face.height * margin);
            
            int x1 = std::max(0, face.x - mx);
            int y1 = std::max(0, face.y - my);
            int x2 = std::min(width, face.x + face.width + mx);
            int y2 = std::min(height, face.y + face.height + my);
            
            int crop_w = x2 - x1;
            int crop_h = y2 - y1;
            
            if (crop_w > 0 && crop_h > 0) {
                int out_w = output_size > 0 ? output_size : crop_w;
                int out_h = output_size > 0 ? output_size : crop_h;
                
                std::vector<float> face_img(out_w * out_h * channels);
                
                for (int oy = 0; oy < out_h; ++oy) {
                    int sy = y1 + oy * crop_h / out_h;
                    for (int ox = 0; ox < out_w; ++ox) {
                        int sx = x1 + ox * crop_w / out_w;
                        for (int c = 0; c < channels; ++c) {
                            face_img[(oy * out_w + ox) * channels + c] = 
                                image[(sy * width + sx) * channels + c];
                        }
                    }
                }
                
                cropped.push_back(std::move(face_img));
            }
        }
        
        return cropped;
    }
    
    Method method() const { return method_; }
    float minConfidence() const { return min_confidence_; }
    void setMinConfidence(float conf) { min_confidence_ = conf; }
    
private:
    Method method_;
    float min_confidence_;
    std::unique_ptr<FaceDetectorBase> detector_;
};

} // namespace face
} // namespace neurova

#endif // NEUROVA_FACE_DETECTOR_HPP
