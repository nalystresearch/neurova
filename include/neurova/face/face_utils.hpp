// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file face_utils.hpp
 * @brief Face processing utilities
 */

#ifndef NEUROVA_FACE_UTILS_HPP
#define NEUROVA_FACE_UTILS_HPP

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>

#include "face_detector.hpp"

namespace neurova {
namespace face {

// ============================================================================
// Image Utilities
// ============================================================================

/**
 * @brief Convert image to grayscale
 */
inline std::vector<float> toGrayscale(
    const float* image, int width, int height, int channels
) {
    std::vector<float> gray(width * height);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            if (channels >= 3) {
                gray[idx] = 0.299f * image[idx * channels + 0] +
                            0.587f * image[idx * channels + 1] +
                            0.114f * image[idx * channels + 2];
            } else {
                gray[idx] = image[idx * channels];
            }
        }
    }
    
    return gray;
}

/**
 * @brief Resize image using bilinear interpolation
 */
inline std::vector<float> resizeImage(
    const float* image, int src_w, int src_h, int channels,
    int dst_w, int dst_h
) {
    std::vector<float> result(dst_w * dst_h * channels);
    
    float x_ratio = static_cast<float>(src_w) / dst_w;
    float y_ratio = static_cast<float>(src_h) / dst_h;
    
    for (int y = 0; y < dst_h; ++y) {
        float src_y = y * y_ratio;
        int y0 = static_cast<int>(src_y);
        int y1 = std::min(y0 + 1, src_h - 1);
        float dy = src_y - y0;
        
        for (int x = 0; x < dst_w; ++x) {
            float src_x = x * x_ratio;
            int x0 = static_cast<int>(src_x);
            int x1 = std::min(x0 + 1, src_w - 1);
            float dx = src_x - x0;
            
            for (int c = 0; c < channels; ++c) {
                float v00 = image[(y0 * src_w + x0) * channels + c];
                float v01 = image[(y0 * src_w + x1) * channels + c];
                float v10 = image[(y1 * src_w + x0) * channels + c];
                float v11 = image[(y1 * src_w + x1) * channels + c];
                
                float v = v00 * (1 - dx) * (1 - dy) +
                          v01 * dx * (1 - dy) +
                          v10 * (1 - dx) * dy +
                          v11 * dx * dy;
                
                result[(y * dst_w + x) * channels + c] = v;
            }
        }
    }
    
    return result;
}

// ============================================================================
// Face Processing
// ============================================================================

/**
 * @brief Crop face from image with margin
 */
inline std::vector<float> cropFace(
    const float* image, int width, int height, int channels,
    const FaceRect& face,
    float margin = 0.0f,
    int output_size = 0
) {
    int mx = static_cast<int>(face.width * margin);
    int my = static_cast<int>(face.height * margin);
    
    int x1 = std::max(0, face.x - mx);
    int y1 = std::max(0, face.y - my);
    int x2 = std::min(width, face.x + face.width + mx);
    int y2 = std::min(height, face.y + face.height + my);
    
    int crop_w = x2 - x1;
    int crop_h = y2 - y1;
    
    if (crop_w <= 0 || crop_h <= 0) {
        return {};
    }
    
    // Extract crop
    std::vector<float> cropped(crop_w * crop_h * channels);
    for (int y = 0; y < crop_h; ++y) {
        for (int x = 0; x < crop_w; ++x) {
            int src_idx = ((y1 + y) * width + (x1 + x)) * channels;
            int dst_idx = (y * crop_w + x) * channels;
            for (int c = 0; c < channels; ++c) {
                cropped[dst_idx + c] = image[src_idx + c];
            }
        }
    }
    
    // Resize if needed
    if (output_size > 0 && (crop_w != output_size || crop_h != output_size)) {
        return resizeImage(cropped.data(), crop_w, crop_h, channels,
                          output_size, output_size);
    }
    
    return cropped;
}

/**
 * @brief Detect landmarks in face image
 * 
 * Returns 5 basic landmarks: left eye, right eye, nose, left mouth, right mouth
 */
inline std::vector<std::pair<int, int>> detectLandmarks(
    const float* face, int width, int height
) {
    std::vector<std::pair<int, int>> landmarks;
    
    // Heuristic-based landmark estimation
    // Left eye
    landmarks.emplace_back(static_cast<int>(width * 0.3), static_cast<int>(height * 0.35));
    // Right eye
    landmarks.emplace_back(static_cast<int>(width * 0.7), static_cast<int>(height * 0.35));
    // Nose tip
    landmarks.emplace_back(static_cast<int>(width * 0.5), static_cast<int>(height * 0.6));
    // Left mouth corner
    landmarks.emplace_back(static_cast<int>(width * 0.35), static_cast<int>(height * 0.75));
    // Right mouth corner
    landmarks.emplace_back(static_cast<int>(width * 0.65), static_cast<int>(height * 0.75));
    
    return landmarks;
}

/**
 * @brief Align face based on eye positions
 */
inline std::vector<float> alignFace(
    const float* face, int width, int height, int channels,
    int output_width = 256,
    int output_height = 256
) {
    auto landmarks = detectLandmarks(face, width, height);
    
    if (landmarks.size() < 2) {
        // Just resize if no landmarks
        return resizeImage(face, width, height, channels, output_width, output_height);
    }
    
    auto [left_eye_x, left_eye_y] = landmarks[0];
    auto [right_eye_x, right_eye_y] = landmarks[1];
    
    // Calculate angle
    float dx = static_cast<float>(right_eye_x - left_eye_x);
    float dy = static_cast<float>(right_eye_y - left_eye_y);
    float angle = std::atan2(dy, dx);
    
    // Calculate scale
    float dist = std::sqrt(dx * dx + dy * dy);
    float desired_dist = output_width * 0.3f;  // Eyes span 30% of output width
    float scale = desired_dist / (dist + 1e-6f);
    
    // Center between eyes
    float cx = (left_eye_x + right_eye_x) / 2.0f;
    float cy = (left_eye_y + right_eye_y) / 2.0f;
    
    // Output center
    float out_cx = output_width / 2.0f;
    float out_cy = output_height * 0.35f;  // Eyes at 35% height
    
    // Transformation
    float cos_a = std::cos(-angle) * scale;
    float sin_a = std::sin(-angle) * scale;
    
    std::vector<float> result(output_width * output_height * channels, 0.0f);
    
    for (int oy = 0; oy < output_height; ++oy) {
        for (int ox = 0; ox < output_width; ++ox) {
            // Map back to source
            float dx = ox - out_cx;
            float dy = oy - out_cy;
            
            float sx = cos_a * dx + sin_a * dy + cx;
            float sy = -sin_a * dx + cos_a * dy + cy;
            
            // Bilinear interpolation
            int x0 = static_cast<int>(sx);
            int y0 = static_cast<int>(sy);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            if (x0 >= 0 && x1 < width && y0 >= 0 && y1 < height) {
                float fx = sx - x0;
                float fy = sy - y0;
                
                for (int c = 0; c < channels; ++c) {
                    float v00 = face[(y0 * width + x0) * channels + c];
                    float v01 = face[(y0 * width + x1) * channels + c];
                    float v10 = face[(y1 * width + x0) * channels + c];
                    float v11 = face[(y1 * width + x1) * channels + c];
                    
                    float v = v00 * (1 - fx) * (1 - fy) +
                              v01 * fx * (1 - fy) +
                              v10 * (1 - fx) * fy +
                              v11 * fx * fy;
                    
                    result[(oy * output_width + ox) * channels + c] = v;
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief Preprocess face for recognition
 */
inline std::vector<float> preprocessFace(
    const float* face, int width, int height, int channels,
    int output_size = 128,
    bool normalize = true
) {
    // Convert to grayscale
    auto gray = toGrayscale(face, width, height, channels);
    
    // Resize
    std::vector<float> resized;
    if (width != output_size || height != output_size) {
        resized = resizeImage(gray.data(), width, height, 1, output_size, output_size);
    } else {
        resized = std::move(gray);
    }
    
    // Normalize
    if (normalize) {
        float mean = 0.0f, std_dev = 0.0f;
        for (float v : resized) mean += v;
        mean /= resized.size();
        
        for (float v : resized) {
            float diff = v - mean;
            std_dev += diff * diff;
        }
        std_dev = std::sqrt(std_dev / resized.size() + 1e-10f);
        
        for (float& v : resized) {
            v = (v - mean) / std_dev;
        }
    }
    
    return resized;
}

// ============================================================================
// Visualization
// ============================================================================

/**
 * @brief Draw rectangle on image
 */
inline void drawRectangle(
    float* image, int width, int height, int channels,
    int x1, int y1, int x2, int y2,
    const float color[3], int thickness = 2
) {
    // Clamp coordinates
    x1 = std::max(0, std::min(x1, width - 1));
    x2 = std::max(0, std::min(x2, width - 1));
    y1 = std::max(0, std::min(y1, height - 1));
    y2 = std::max(0, std::min(y2, height - 1));
    
    // Draw horizontal lines
    for (int t = 0; t < thickness; ++t) {
        // Top
        if (y1 + t < height) {
            for (int x = x1; x <= x2; ++x) {
                for (int c = 0; c < std::min(channels, 3); ++c) {
                    image[((y1 + t) * width + x) * channels + c] = color[c];
                }
            }
        }
        // Bottom
        if (y2 - t >= 0) {
            for (int x = x1; x <= x2; ++x) {
                for (int c = 0; c < std::min(channels, 3); ++c) {
                    image[((y2 - t) * width + x) * channels + c] = color[c];
                }
            }
        }
    }
    
    // Draw vertical lines
    for (int t = 0; t < thickness; ++t) {
        // Left
        if (x1 + t < width) {
            for (int y = y1; y <= y2; ++y) {
                for (int c = 0; c < std::min(channels, 3); ++c) {
                    image[(y * width + x1 + t) * channels + c] = color[c];
                }
            }
        }
        // Right
        if (x2 - t >= 0) {
            for (int y = y1; y <= y2; ++y) {
                for (int c = 0; c < std::min(channels, 3); ++c) {
                    image[(y * width + x2 - t) * channels + c] = color[c];
                }
            }
        }
    }
}

/**
 * @brief Draw detected faces on image
 */
inline void drawFaces(
    float* image, int width, int height, int channels,
    const std::vector<FaceRect>& faces,
    const float color[3] = nullptr,
    int thickness = 2
) {
    float default_color[3] = {0.0f, 255.0f, 0.0f}; // Green
    const float* c = color ? color : default_color;
    
    for (const auto& face : faces) {
        drawRectangle(image, width, height, channels,
                     face.x, face.y,
                     face.x + face.width, face.y + face.height,
                     c, thickness);
    }
}

// ============================================================================
// Face Verification
// ============================================================================

/**
 * @brief Compute distance between two face embeddings
 */
inline float computeFaceDistance(
    const std::vector<float>& embedding1,
    const std::vector<float>& embedding2
) {
    if (embedding1.size() != embedding2.size()) {
        return std::numeric_limits<float>::infinity();
    }
    
    // Cosine distance
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (size_t i = 0; i < embedding1.size(); ++i) {
        dot += embedding1[i] * embedding2[i];
        norm1 += embedding1[i] * embedding1[i];
        norm2 += embedding2[i] * embedding2[i];
    }
    
    return 1.0f - dot / (std::sqrt(norm1 * norm2) + 1e-10f);
}

/**
 * @brief Verify if two faces match
 */
inline bool verifyFaces(
    const std::vector<float>& embedding1,
    const std::vector<float>& embedding2,
    float threshold = 0.6f
) {
    return computeFaceDistance(embedding1, embedding2) < threshold;
}

// ============================================================================
// Data Augmentation
// ============================================================================

/**
 * @brief Flip face horizontally
 */
inline std::vector<float> flipHorizontal(
    const float* image, int width, int height, int channels
) {
    std::vector<float> result(width * height * channels);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int src_x = width - 1 - x;
            for (int c = 0; c < channels; ++c) {
                result[(y * width + x) * channels + c] = 
                    image[(y * width + src_x) * channels + c];
            }
        }
    }
    
    return result;
}

/**
 * @brief Adjust brightness
 */
inline std::vector<float> adjustBrightness(
    const float* image, int width, int height, int channels,
    float factor
) {
    std::vector<float> result(width * height * channels);
    
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = std::max(0.0f, std::min(255.0f, image[i] * factor));
    }
    
    return result;
}

/**
 * @brief Add Gaussian noise
 */
inline std::vector<float> addNoise(
    const float* image, int width, int height, int channels,
    float sigma = 10.0f
) {
    std::vector<float> result(width * height * channels);
    
    // Simple pseudo-random noise (for demo purposes)
    unsigned int seed = 42;
    for (size_t i = 0; i < result.size(); ++i) {
        seed = seed * 1103515245 + 12345;
        float noise = (static_cast<float>(seed % 1000) / 500.0f - 1.0f) * sigma;
        result[i] = std::max(0.0f, std::min(255.0f, image[i] + noise));
    }
    
    return result;
}

} // namespace face
} // namespace neurova

#endif // NEUROVA_FACE_UTILS_HPP
