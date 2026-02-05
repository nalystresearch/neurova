// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file blazeface_detector.hpp
 * @brief BlazeFace-style face detector using anchor-based detection
 */

#ifndef NEUROVA_FACE_BLAZEFACE_DETECTOR_HPP
#define NEUROVA_FACE_BLAZEFACE_DETECTOR_HPP

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace neurova {
namespace face {

// ============================================================================
// Anchor Generation
// ============================================================================

/**
 * @brief Anchor box structure
 */
struct Anchor {
    float y_center;
    float x_center;
    float height;
    float width;
};

/**
 * @brief Generate SSD-style anchors for face detection
 */
inline std::vector<Anchor> generateAnchors(
    int input_size = 128,
    const std::vector<int>& strides = {8, 16, 16, 16}
) {
    std::vector<Anchor> anchors;
    
    for (size_t layer = 0; layer < strides.size(); ++layer) {
        int stride = strides[layer];
        int feature_size = input_size / stride;
        
        for (int y = 0; y < feature_size; ++y) {
            for (int x = 0; x < feature_size; ++x) {
                float cx = (x + 0.5f) / feature_size;
                float cy = (y + 0.5f) / feature_size;
                
                // Two anchors per position
                anchors.push_back({cy, cx, 1.0f, 1.0f});
                anchors.push_back({cy, cx, 1.0f, 1.0f});
            }
        }
    }
    
    return anchors;
}

// ============================================================================
// Neural Network Primitives
// ============================================================================

/**
 * @brief 2D convolution
 */
inline std::vector<float> conv2d(
    const float* input, int height, int width,
    const float* kernel, int kh, int kw,
    int stride = 1, int padding = 0
) {
    int padded_h = height + 2 * padding;
    int padded_w = width + 2 * padding;
    
    // Create padded input
    std::vector<float> padded(padded_h * padded_w, 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            padded[(y + padding) * padded_w + (x + padding)] = input[y * width + x];
        }
    }
    
    int oh = (padded_h - kh) / stride + 1;
    int ow = (padded_w - kw) / stride + 1;
    
    std::vector<float> output(oh * ow, 0.0f);
    
    for (int oy = 0; oy < oh; ++oy) {
        for (int ox = 0; ox < ow; ++ox) {
            float sum = 0.0f;
            for (int ky = 0; ky < kh; ++ky) {
                for (int kx = 0; kx < kw; ++kx) {
                    int iy = oy * stride + ky;
                    int ix = ox * stride + kx;
                    sum += padded[iy * padded_w + ix] * kernel[ky * kw + kx];
                }
            }
            output[oy * ow + ox] = sum;
        }
    }
    
    return output;
}

/**
 * @brief ReLU activation
 */
inline void relu(float* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

/**
 * @brief Sigmoid activation
 */
inline float sigmoid(float x) {
    x = std::max(-50.0f, std::min(50.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

// ============================================================================
// Box Decoding
// ============================================================================

/**
 * @brief Decoded bounding box
 */
struct DecodedBox {
    float ymin, xmin, ymax, xmax;
    float confidence;
};

/**
 * @brief Decode raw box predictions using anchors
 */
inline std::vector<DecodedBox> decodeBoxes(
    const std::vector<float>& raw_boxes,
    const std::vector<float>& scores,
    const std::vector<Anchor>& anchors,
    float scale = 128.0f
) {
    std::vector<DecodedBox> boxes;
    size_t n = anchors.size();
    
    for (size_t i = 0; i < n; ++i) {
        float y_center = raw_boxes[i * 4 + 0] / scale * anchors[i].height + anchors[i].y_center;
        float x_center = raw_boxes[i * 4 + 1] / scale * anchors[i].width + anchors[i].x_center;
        float h = raw_boxes[i * 4 + 2] / scale * anchors[i].height;
        float w = raw_boxes[i * 4 + 3] / scale * anchors[i].width;
        
        DecodedBox box;
        box.ymin = y_center - h / 2;
        box.xmin = x_center - w / 2;
        box.ymax = y_center + h / 2;
        box.xmax = x_center + w / 2;
        box.confidence = scores[i];
        boxes.push_back(box);
    }
    
    return boxes;
}

// ============================================================================
// Non-Maximum Suppression
// ============================================================================

/**
 * @brief Compute IoU between two boxes
 */
inline float computeIoU(const DecodedBox& a, const DecodedBox& b) {
    float y1 = std::max(a.ymin, b.ymin);
    float x1 = std::max(a.xmin, b.xmin);
    float y2 = std::min(a.ymax, b.ymax);
    float x2 = std::min(a.xmax, b.xmax);
    
    float inter = std::max(0.0f, y2 - y1) * std::max(0.0f, x2 - x1);
    float area_a = (a.ymax - a.ymin) * (a.xmax - a.xmin);
    float area_b = (b.ymax - b.ymin) * (b.xmax - b.xmin);
    
    return inter / (area_a + area_b - inter + 1e-6f);
}

/**
 * @brief Non-maximum suppression
 */
inline std::vector<size_t> nms(
    const std::vector<DecodedBox>& boxes,
    float iou_threshold = 0.3f,
    float score_threshold = 0.5f
) {
    std::vector<size_t> keep;
    
    // Filter by score
    std::vector<std::pair<float, size_t>> scored;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (boxes[i].confidence > score_threshold) {
            scored.emplace_back(boxes[i].confidence, i);
        }
    }
    
    if (scored.empty()) return keep;
    
    // Sort by score
    std::sort(scored.begin(), scored.end(), std::greater<>());
    
    std::vector<bool> suppressed(scored.size(), false);
    
    for (size_t i = 0; i < scored.size(); ++i) {
        if (suppressed[i]) continue;
        
        size_t idx = scored[i].second;
        keep.push_back(idx);
        
        for (size_t j = i + 1; j < scored.size(); ++j) {
            if (suppressed[j]) continue;
            
            size_t jdx = scored[j].second;
            if (computeIoU(boxes[idx], boxes[jdx]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return keep;
}

// ============================================================================
// BlazeFace Detector
// ============================================================================

/**
 * @brief BlazeFace detection result
 */
struct BlazeFaceDetection {
    int x, y, width, height;
    float confidence;
};

/**
 * @brief BlazeFace-style face detector
 * 
 * Pure C++ implementation using convolution-based feature extraction
 * and anchor-based detection.
 */
class BlazeFaceDetector {
public:
    static constexpr int INPUT_SIZE = 128;
    
    /**
     * @brief Construct detector
     */
    BlazeFaceDetector(
        float min_confidence = 0.5f,
        float nms_threshold = 0.3f
    ) : min_confidence_(min_confidence),
        nms_threshold_(nms_threshold) {
        initKernels();
        anchors_ = generateAnchors(INPUT_SIZE);
    }
    
    /**
     * @brief Detect faces in image
     * 
     * @param image Grayscale image data
     * @param width Image width
     * @param height Image height
     * @param min_size Minimum face size
     * @return Vector of face detections
     */
    std::vector<BlazeFaceDetection> detect(
        const float* image, int width, int height,
        int min_size = 20
    ) const {
        int orig_w = width;
        int orig_h = height;
        
        // Preprocess (resize to INPUT_SIZE)
        auto preprocessed = preprocess(image, width, height);
        
        // Extract features
        auto features = extractFeatures(preprocessed.data(), INPUT_SIZE, INPUT_SIZE);
        
        // Detect faces
        auto detections = detectFaces(features, orig_w, orig_h);
        
        // Filter by minimum size
        std::vector<BlazeFaceDetection> filtered;
        for (const auto& det : detections) {
            if (det.width >= min_size && det.height >= min_size) {
                filtered.push_back(det);
            }
        }
        
        return nmsFilter(filtered);
    }

private:
    float min_confidence_;
    float nms_threshold_;
    std::vector<Anchor> anchors_;
    
    // Convolution kernels
    std::array<float, 9> kernel_sobel_h_;
    std::array<float, 9> kernel_sobel_v_;
    std::array<float, 9> kernel_blur_;
    
    void initKernels() {
        // Sobel horizontal
        kernel_sobel_h_ = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        // Sobel vertical
        kernel_sobel_v_ = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
        // Gaussian blur
        kernel_blur_ = {
            1/16.0f, 2/16.0f, 1/16.0f,
            2/16.0f, 4/16.0f, 2/16.0f,
            1/16.0f, 2/16.0f, 1/16.0f
        };
    }
    
    std::vector<float> preprocess(const float* image, int width, int height) const {
        std::vector<float> result(INPUT_SIZE * INPUT_SIZE);
        
        float x_ratio = static_cast<float>(width) / INPUT_SIZE;
        float y_ratio = static_cast<float>(height) / INPUT_SIZE;
        
        for (int y = 0; y < INPUT_SIZE; ++y) {
            for (int x = 0; x < INPUT_SIZE; ++x) {
                float src_x = x * x_ratio;
                float src_y = y * y_ratio;
                int sx = static_cast<int>(src_x);
                int sy = static_cast<int>(src_y);
                
                sx = std::min(sx, width - 1);
                sy = std::min(sy, height - 1);
                
                // Normalize to [0, 1]
                result[y * INPUT_SIZE + x] = image[sy * width + sx] / 255.0f;
            }
        }
        
        return result;
    }
    
    struct Features {
        std::vector<float> intensity;
        std::vector<float> edges;
        std::vector<float> smooth;
        int width, height;
    };
    
    Features extractFeatures(const float* image, int width, int height) const {
        Features f;
        f.width = width;
        f.height = height;
        f.intensity.assign(image, image + width * height);
        
        // Edge detection
        auto edges_h = conv2d(image, height, width, kernel_sobel_h_.data(), 3, 3, 1, 1);
        auto edges_v = conv2d(image, height, width, kernel_sobel_v_.data(), 3, 3, 1, 1);
        
        f.edges.resize(width * height);
        for (size_t i = 0; i < f.edges.size(); ++i) {
            f.edges[i] = std::sqrt(edges_h[i] * edges_h[i] + edges_v[i] * edges_v[i]);
        }
        
        // Smoothing
        f.smooth = conv2d(image, height, width, kernel_blur_.data(), 3, 3, 1, 1);
        
        return f;
    }
    
    std::vector<BlazeFaceDetection> detectFaces(
        const Features& features,
        int orig_w, int orig_h
    ) const {
        std::vector<BlazeFaceDetection> faces;
        
        // Multi-scale sliding window detection
        std::vector<int> window_sizes = {32, 24, 16};
        
        for (int win_size : window_sizes) {
            int stride = win_size / 2;
            
            for (int y = 0; y <= features.height - win_size; y += stride) {
                for (int x = 0; x <= features.width - win_size; x += stride) {
                    float score = computeFaceScore(features, x, y, win_size);
                    
                    if (score >= min_confidence_) {
                        BlazeFaceDetection det;
                        det.x = static_cast<int>(static_cast<float>(x) / INPUT_SIZE * orig_w);
                        det.y = static_cast<int>(static_cast<float>(y) / INPUT_SIZE * orig_h);
                        det.width = static_cast<int>(static_cast<float>(win_size) / INPUT_SIZE * orig_w);
                        det.height = static_cast<int>(static_cast<float>(win_size) / INPUT_SIZE * orig_h);
                        det.confidence = score;
                        faces.push_back(det);
                    }
                }
            }
        }
        
        return faces;
    }
    
    float computeFaceScore(
        const Features& f, int x, int y, int win_size
    ) const {
        // Extract window statistics
        float intensity_sum = 0.0f, edge_sum = 0.0f;
        int count = 0;
        
        for (int wy = 0; wy < win_size && (y + wy) < f.height; ++wy) {
            for (int wx = 0; wx < win_size && (x + wx) < f.width; ++wx) {
                int idx = (y + wy) * f.width + (x + wx);
                intensity_sum += f.intensity[idx];
                edge_sum += f.edges[idx];
                ++count;
            }
        }
        
        if (count == 0) return 0.0f;
        
        float mean_intensity = intensity_sum / count;
        float mean_edge = edge_sum / count;
        
        // Compute symmetry score
        float symmetry = computeSymmetry(f.intensity.data(), f.width, x, y, win_size);
        
        // Face heuristics
        float brightness_score = (mean_intensity > 0.3f && mean_intensity < 0.8f) ? 1.0f : 0.5f;
        float edge_score = std::min(1.0f, mean_edge * 5.0f);
        
        // Combine scores
        float score = symmetry * 0.35f + brightness_score * 0.3f + edge_score * 0.35f;
        
        return std::max(0.0f, std::min(1.0f, score));
    }
    
    float computeSymmetry(
        const float* data, int stride,
        int x, int y, int win_size
    ) const {
        float diff_sum = 0.0f;
        int count = 0;
        int half = win_size / 2;
        
        for (int wy = 0; wy < win_size; ++wy) {
            for (int wx = 0; wx < half; ++wx) {
                int left_idx = (y + wy) * stride + (x + wx);
                int right_idx = (y + wy) * stride + (x + win_size - 1 - wx);
                
                diff_sum += std::abs(data[left_idx] - data[right_idx]);
                ++count;
            }
        }
        
        if (count == 0) return 0.5f;
        return 1.0f - diff_sum / count;
    }
    
    std::vector<BlazeFaceDetection> nmsFilter(
        std::vector<BlazeFaceDetection>& detections
    ) const {
        if (detections.empty()) return {};
        
        // Sort by confidence
        std::sort(detections.begin(), detections.end(),
                  [](const auto& a, const auto& b) {
                      return a.confidence > b.confidence;
                  });
        
        std::vector<BlazeFaceDetection> keep;
        std::vector<bool> suppressed(detections.size(), false);
        
        for (size_t i = 0; i < detections.size(); ++i) {
            if (suppressed[i]) continue;
            
            keep.push_back(detections[i]);
            
            for (size_t j = i + 1; j < detections.size(); ++j) {
                if (suppressed[j]) continue;
                
                if (computeIoU(detections[i], detections[j]) > nms_threshold_) {
                    suppressed[j] = true;
                }
            }
        }
        
        return keep;
    }
    
    float computeIoU(
        const BlazeFaceDetection& a,
        const BlazeFaceDetection& b
    ) const {
        int x1 = std::max(a.x, b.x);
        int y1 = std::max(a.y, b.y);
        int x2 = std::min(a.x + a.width, b.x + b.width);
        int y2 = std::min(a.y + a.height, b.y + b.height);
        
        int inter = std::max(0, x2 - x1) * std::max(0, y2 - y1);
        int area_a = a.width * a.height;
        int area_b = b.width * b.height;
        
        return static_cast<float>(inter) / (area_a + area_b - inter + 1e-6f);
    }
};

} // namespace face
} // namespace neurova

#endif // NEUROVA_FACE_BLAZEFACE_DETECTOR_HPP
