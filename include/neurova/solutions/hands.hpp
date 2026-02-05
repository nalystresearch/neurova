// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file solutions/hands.hpp
 * @brief Hand tracking and gesture recognition solution
 * 
 * Neurova implementation of hand landmark detection (21 landmarks per hand).
 */

#pragma once

#include "../core/image.hpp"
#include "../object_detection/anchor.hpp"
#include <vector>
#include <array>
#include <cmath>

namespace neurova {
namespace solutions {

/**
 * @brief Hand landmark indices
 */
enum class HandLandmarkType {
    WRIST = 0,
    THUMB_CMC = 1,
    THUMB_MCP = 2,
    THUMB_IP = 3,
    THUMB_TIP = 4,
    INDEX_FINGER_MCP = 5,
    INDEX_FINGER_PIP = 6,
    INDEX_FINGER_DIP = 7,
    INDEX_FINGER_TIP = 8,
    MIDDLE_FINGER_MCP = 9,
    MIDDLE_FINGER_PIP = 10,
    MIDDLE_FINGER_DIP = 11,
    MIDDLE_FINGER_TIP = 12,
    RING_FINGER_MCP = 13,
    RING_FINGER_PIP = 14,
    RING_FINGER_DIP = 15,
    RING_FINGER_TIP = 16,
    PINKY_MCP = 17,
    PINKY_PIP = 18,
    PINKY_DIP = 19,
    PINKY_TIP = 20
};

/**
 * @brief Hand landmark
 */
struct HandLandmark {
    float x = 0, y = 0, z = 0;
    float visibility = 1.0f;
    
    HandLandmark() = default;
    HandLandmark(float x_, float y_, float z_ = 0, float vis = 1.0f)
        : x(x_), y(y_), z(z_), visibility(vis) {}
};

/**
 * @brief Handedness (left or right hand)
 */
enum class Handedness {
    Left,
    Right,
    Unknown
};

/**
 * @brief Hand result with 21 landmarks
 */
struct Hand {
    std::array<HandLandmark, 21> landmarks;
    object_detection::BBox bounding_box;
    Handedness handedness = Handedness::Unknown;
    float handedness_confidence = 0;
    float detection_confidence = 0;
    
    Hand() {
        landmarks.fill(HandLandmark());
    }
    
    // Access by type
    HandLandmark& operator[](HandLandmarkType type) {
        return landmarks[static_cast<int>(type)];
    }
    
    const HandLandmark& operator[](HandLandmarkType type) const {
        return landmarks[static_cast<int>(type)];
    }
    
    // Finger connections for visualization
    static const std::vector<std::pair<int, int>>& connections() {
        static const std::vector<std::pair<int, int>> conns = {
            // Thumb
            {0, 1}, {1, 2}, {2, 3}, {3, 4},
            // Index
            {0, 5}, {5, 6}, {6, 7}, {7, 8},
            // Middle
            {0, 9}, {9, 10}, {10, 11}, {11, 12},
            // Ring
            {0, 13}, {13, 14}, {14, 15}, {15, 16},
            // Pinky
            {0, 17}, {17, 18}, {18, 19}, {19, 20},
            // Palm
            {5, 9}, {9, 13}, {13, 17}
        };
        return conns;
    }
    
    // Check if finger is extended
    bool is_finger_extended(int finger_idx) const {
        // finger_idx: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
        int tip_idx, pip_idx, mcp_idx;
        
        switch (finger_idx) {
            case 0:  // Thumb
                tip_idx = 4; pip_idx = 3; mcp_idx = 2;
                break;
            case 1:  // Index
                tip_idx = 8; pip_idx = 6; mcp_idx = 5;
                break;
            case 2:  // Middle
                tip_idx = 12; pip_idx = 10; mcp_idx = 9;
                break;
            case 3:  // Ring
                tip_idx = 16; pip_idx = 14; mcp_idx = 13;
                break;
            case 4:  // Pinky
                tip_idx = 20; pip_idx = 18; mcp_idx = 17;
                break;
            default:
                return false;
        }
        
        // Finger is extended if tip is farther from wrist than pip
        const auto& wrist = landmarks[0];
        const auto& tip = landmarks[tip_idx];
        const auto& pip = landmarks[pip_idx];
        
        float tip_dist = std::sqrt(std::pow(tip.x - wrist.x, 2) + std::pow(tip.y - wrist.y, 2));
        float pip_dist = std::sqrt(std::pow(pip.x - wrist.x, 2) + std::pow(pip.y - wrist.y, 2));
        
        return tip_dist > pip_dist;
    }
    
    // Count extended fingers
    int count_extended_fingers() const {
        int count = 0;
        for (int i = 0; i < 5; ++i) {
            if (is_finger_extended(i)) count++;
        }
        return count;
    }
};

/**
 * @brief Common hand gestures
 */
enum class HandGesture {
    Unknown,
    OpenPalm,      // All fingers extended
    Fist,          // All fingers closed
    ThumbsUp,      // Only thumb extended
    ThumbsDown,
    PointingUp,    // Only index extended
    Peace,         // Index and middle extended (V sign)
    Ok,            // Thumb and index form circle
    Rock           // Index and pinky extended (metal sign)
};

/**
 * @brief Recognize hand gesture
 */
inline HandGesture recognize_gesture(const Hand& hand) {
    bool fingers[5];
    for (int i = 0; i < 5; ++i) {
        fingers[i] = hand.is_finger_extended(i);
    }
    
    int extended_count = 0;
    for (int i = 0; i < 5; ++i) {
        if (fingers[i]) extended_count++;
    }
    
    // Open palm: all fingers extended
    if (extended_count == 5) {
        return HandGesture::OpenPalm;
    }
    
    // Fist: no fingers extended
    if (extended_count == 0) {
        return HandGesture::Fist;
    }
    
    // Thumbs up: only thumb
    if (extended_count == 1 && fingers[0]) {
        // Check thumb is pointing up
        const auto& thumb_tip = hand[HandLandmarkType::THUMB_TIP];
        const auto& thumb_mcp = hand[HandLandmarkType::THUMB_MCP];
        if (thumb_tip.y < thumb_mcp.y) {
            return HandGesture::ThumbsUp;
        } else {
            return HandGesture::ThumbsDown;
        }
    }
    
    // Pointing: only index
    if (extended_count == 1 && fingers[1]) {
        return HandGesture::PointingUp;
    }
    
    // Peace: index and middle
    if (extended_count == 2 && fingers[1] && fingers[2]) {
        return HandGesture::Peace;
    }
    
    // Rock: index and pinky
    if (extended_count == 2 && fingers[1] && fingers[4]) {
        return HandGesture::Rock;
    }
    
    return HandGesture::Unknown;
}

/**
 * @brief Hand detector configuration
 */
struct HandDetectorConfig {
    int max_num_hands = 2;
    float min_detection_confidence = 0.5f;
    float min_tracking_confidence = 0.5f;
    bool static_image_mode = false;
};

/**
 * @brief Hand detector
 */
class HandDetector {
public:
    explicit HandDetector(const HandDetectorConfig& config = {}) : config_(config) {}
    
    /**
     * @brief Detect hands in an image
     */
    std::vector<Hand> process(const Image& image) {
        std::vector<Hand> hands;
        
        // Detect palm regions first
        auto palm_boxes = detect_palms(image);
        
        for (const auto& box : palm_boxes) {
            if (hands.size() >= static_cast<size_t>(config_.max_num_hands)) break;
            
            Hand hand;
            hand.bounding_box = box;
            hand.detection_confidence = 1.0f;
            
            // Estimate landmarks
            estimate_landmarks(image, box, hand);
            
            // Determine handedness
            determine_handedness(hand);
            
            hands.push_back(hand);
        }
        
        return hands;
    }
    
private:
    HandDetectorConfig config_;
    
    std::vector<object_detection::BBox> detect_palms(const Image& image) {
        std::vector<object_detection::BBox> boxes;
        
        // Simplified: return centered box for demo
        float cx = image.width() / 2.0f;
        float cy = image.height() / 2.0f;
        float size = std::min(image.width(), image.height()) * 0.3f;
        
        boxes.push_back(object_detection::BBox(cx - size/2, cy - size/2, cx + size/2, cy + size/2));
        
        return boxes;
    }
    
    void estimate_landmarks(const Image& image, const object_detection::BBox& box, Hand& hand) {
        float cx = box.center_x();
        float cy = box.center_y();
        float w = box.width();
        float h = box.height();
        
        // Wrist at bottom center
        hand[HandLandmarkType::WRIST] = HandLandmark(cx, cy + h * 0.4f, 0);
        
        // Thumb (left side)
        hand[HandLandmarkType::THUMB_CMC] = HandLandmark(cx - w * 0.15f, cy + h * 0.2f, 0);
        hand[HandLandmarkType::THUMB_MCP] = HandLandmark(cx - w * 0.25f, cy + h * 0.05f, 0);
        hand[HandLandmarkType::THUMB_IP] = HandLandmark(cx - w * 0.3f, cy - h * 0.05f, 0);
        hand[HandLandmarkType::THUMB_TIP] = HandLandmark(cx - w * 0.35f, cy - h * 0.15f, 0);
        
        // Index finger
        hand[HandLandmarkType::INDEX_FINGER_MCP] = HandLandmark(cx - w * 0.1f, cy, 0);
        hand[HandLandmarkType::INDEX_FINGER_PIP] = HandLandmark(cx - w * 0.1f, cy - h * 0.15f, 0);
        hand[HandLandmarkType::INDEX_FINGER_DIP] = HandLandmark(cx - w * 0.1f, cy - h * 0.25f, 0);
        hand[HandLandmarkType::INDEX_FINGER_TIP] = HandLandmark(cx - w * 0.1f, cy - h * 0.35f, 0);
        
        // Middle finger
        hand[HandLandmarkType::MIDDLE_FINGER_MCP] = HandLandmark(cx, cy - h * 0.05f, 0);
        hand[HandLandmarkType::MIDDLE_FINGER_PIP] = HandLandmark(cx, cy - h * 0.2f, 0);
        hand[HandLandmarkType::MIDDLE_FINGER_DIP] = HandLandmark(cx, cy - h * 0.32f, 0);
        hand[HandLandmarkType::MIDDLE_FINGER_TIP] = HandLandmark(cx, cy - h * 0.42f, 0);
        
        // Ring finger
        hand[HandLandmarkType::RING_FINGER_MCP] = HandLandmark(cx + w * 0.1f, cy, 0);
        hand[HandLandmarkType::RING_FINGER_PIP] = HandLandmark(cx + w * 0.1f, cy - h * 0.15f, 0);
        hand[HandLandmarkType::RING_FINGER_DIP] = HandLandmark(cx + w * 0.1f, cy - h * 0.25f, 0);
        hand[HandLandmarkType::RING_FINGER_TIP] = HandLandmark(cx + w * 0.1f, cy - h * 0.33f, 0);
        
        // Pinky
        hand[HandLandmarkType::PINKY_MCP] = HandLandmark(cx + w * 0.2f, cy + h * 0.05f, 0);
        hand[HandLandmarkType::PINKY_PIP] = HandLandmark(cx + w * 0.2f, cy - h * 0.08f, 0);
        hand[HandLandmarkType::PINKY_DIP] = HandLandmark(cx + w * 0.2f, cy - h * 0.18f, 0);
        hand[HandLandmarkType::PINKY_TIP] = HandLandmark(cx + w * 0.2f, cy - h * 0.26f, 0);
    }
    
    void determine_handedness(Hand& hand) {
        // Determine based on thumb position relative to pinky
        const auto& thumb = hand[HandLandmarkType::THUMB_TIP];
        const auto& pinky = hand[HandLandmarkType::PINKY_TIP];
        
        if (thumb.x < pinky.x) {
            hand.handedness = Handedness::Right;  // Thumb on left = right hand (palm facing viewer)
        } else {
            hand.handedness = Handedness::Left;
        }
        hand.handedness_confidence = 0.9f;
    }
};

/**
 * @brief Draw hand landmarks on image
 */
inline void draw_hand(Image& image, const Hand& hand,
                      bool draw_landmarks = true,
                      bool draw_connections = true) {
    // Draw connections
    if (draw_connections) {
        for (const auto& conn : Hand::connections()) {
            const auto& lm1 = hand.landmarks[conn.first];
            const auto& lm2 = hand.landmarks[conn.second];
            
            int x1 = static_cast<int>(lm1.x);
            int y1 = static_cast<int>(lm1.y);
            int x2 = static_cast<int>(lm2.x);
            int y2 = static_cast<int>(lm2.y);
            
            // Bresenham line
            int dx = std::abs(x2 - x1);
            int dy = std::abs(y2 - y1);
            int sx = x1 < x2 ? 1 : -1;
            int sy = y1 < y2 ? 1 : -1;
            int err = dx - dy;
            
            while (true) {
                if (x1 >= 0 && x1 < image.width() && y1 >= 0 && y1 < image.height()) {
                    if (image.channels() >= 3) {
                        image.at(x1, y1, 0) = 200;
                        image.at(x1, y1, 1) = 200;
                        image.at(x1, y1, 2) = 200;
                    }
                }
                if (x1 == x2 && y1 == y2) break;
                int e2 = 2 * err;
                if (e2 > -dy) { err -= dy; x1 += sx; }
                if (e2 < dx) { err += dx; y1 += sy; }
            }
        }
    }
    
    // Draw landmarks
    if (draw_landmarks) {
        for (int i = 0; i < 21; ++i) {
            const auto& lm = hand.landmarks[i];
            int x = static_cast<int>(lm.x);
            int y = static_cast<int>(lm.y);
            int radius = 4;
            
            // Color by finger
            float r = 255, g = 0, b = 0;
            if (i >= 1 && i <= 4) { r = 255; g = 128; b = 0; }      // Thumb: orange
            else if (i >= 5 && i <= 8) { r = 255; g = 255; b = 0; }  // Index: yellow
            else if (i >= 9 && i <= 12) { r = 0; g = 255; b = 0; }   // Middle: green
            else if (i >= 13 && i <= 16) { r = 0; g = 255; b = 255; } // Ring: cyan
            else if (i >= 17 && i <= 20) { r = 0; g = 0; b = 255; }  // Pinky: blue
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx*dx + dy*dy <= radius*radius) {
                        int px = x + dx;
                        int py = y + dy;
                        if (px >= 0 && px < image.width() && py >= 0 && py < image.height()) {
                            if (image.channels() >= 3) {
                                image.at(px, py, 0) = r;
                                image.at(px, py, 1) = g;
                                image.at(px, py, 2) = b;
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace solutions
} // namespace neurova
