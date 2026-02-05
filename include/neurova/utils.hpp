// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file utils.hpp
 * @brief Drawing and utility functions
 */

#ifndef NEUROVA_UTILS_HPP
#define NEUROVA_UTILS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <array>
#include <random>
#include <string>

namespace neurova {
namespace utils {

// ============================================================================
// Color Utilities
// ============================================================================

struct Color {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t a = 255;
    
    Color() = default;
    Color(uint8_t r_, uint8_t g_, uint8_t b_, uint8_t a_ = 255) 
        : r(r_), g(g_), b(b_), a(a_) {}
    
    static Color Red() { return Color(255, 0, 0); }
    static Color Green() { return Color(0, 255, 0); }
    static Color Blue() { return Color(0, 0, 255); }
    static Color White() { return Color(255, 255, 255); }
    static Color Black() { return Color(0, 0, 0); }
    static Color Yellow() { return Color(255, 255, 0); }
    static Color Cyan() { return Color(0, 255, 255); }
    static Color Magenta() { return Color(255, 0, 255); }
    static Color Orange() { return Color(255, 165, 0); }
};

// ============================================================================
// Point Structure
// ============================================================================

struct Point2i {
    int x = 0;
    int y = 0;
    
    Point2i() = default;
    Point2i(int x_, int y_) : x(x_), y(y_) {}
};

struct Point2f {
    float x = 0;
    float y = 0;
    
    Point2f() = default;
    Point2f(float x_, float y_) : x(x_), y(y_) {}
};

// ============================================================================
// Keypoint Structure (for drawing)
// ============================================================================

struct DrawKeypoint {
    float x = 0;
    float y = 0;
    float size = 1;
    float angle = -1;
    float response = 0;
    int octave = 0;
    
    DrawKeypoint() = default;
    DrawKeypoint(float x_, float y_, float size_ = 1, float angle_ = -1, 
                float response_ = 0, int octave_ = 0)
        : x(x_), y(y_), size(size_), angle(angle_), response(response_), octave(octave_) {}
};

// ============================================================================
// Match Structure
// ============================================================================

struct DrawMatch {
    int queryIdx = 0;
    int trainIdx = 0;
    float distance = 0;
    
    DrawMatch() = default;
    DrawMatch(int q, int t, float d = 0) : queryIdx(q), trainIdx(t), distance(d) {}
};

// ============================================================================
// Drawing Primitives
// ============================================================================

/**
 * @brief Set pixel with bounds checking
 */
inline void setPixel(uint8_t* img, int w, int h, int channels,
                    int x, int y, const Color& color) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    
    int idx = (y * w + x) * channels;
    if (channels >= 1) img[idx] = color.r;
    if (channels >= 2) img[idx + 1] = color.g;
    if (channels >= 3) img[idx + 2] = color.b;
    if (channels >= 4) img[idx + 3] = color.a;
}

/**
 * @brief Draw a line (Bresenham's algorithm)
 */
inline void drawLine(uint8_t* img, int w, int h, int channels,
                    int x1, int y1, int x2, int y2,
                    const Color& color, int thickness = 1) {
    int dx = std::abs(x2 - x1);
    int dy = std::abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;
    
    int halfThick = thickness / 2;
    
    while (true) {
        // Draw thick point
        for (int tx = -halfThick; tx <= halfThick; ++tx) {
            for (int ty = -halfThick; ty <= halfThick; ++ty) {
                setPixel(img, w, h, channels, x1 + tx, y1 + ty, color);
            }
        }
        
        if (x1 == x2 && y1 == y2) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y1 += sy;
        }
    }
}

/**
 * @brief Draw a rectangle
 */
inline void drawRectangle(uint8_t* img, int w, int h, int channels,
                         int x1, int y1, int x2, int y2,
                         const Color& color, int thickness = 1, bool filled = false) {
    if (filled) {
        int minX = std::max(0, std::min(x1, x2));
        int maxX = std::min(w - 1, std::max(x1, x2));
        int minY = std::max(0, std::min(y1, y2));
        int maxY = std::min(h - 1, std::max(y1, y2));
        
        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                setPixel(img, w, h, channels, x, y, color);
            }
        }
    } else {
        drawLine(img, w, h, channels, x1, y1, x2, y1, color, thickness);
        drawLine(img, w, h, channels, x2, y1, x2, y2, color, thickness);
        drawLine(img, w, h, channels, x2, y2, x1, y2, color, thickness);
        drawLine(img, w, h, channels, x1, y2, x1, y1, color, thickness);
    }
}

/**
 * @brief Draw a circle (Midpoint algorithm)
 */
inline void drawCircle(uint8_t* img, int w, int h, int channels,
                      int cx, int cy, int radius,
                      const Color& color, int thickness = 1, bool filled = false) {
    if (filled) {
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                if (x * x + y * y <= radius * radius) {
                    setPixel(img, w, h, channels, cx + x, cy + y, color);
                }
            }
        }
    } else {
        int x = radius;
        int y = 0;
        int err = 0;
        int halfThick = thickness / 2;
        
        while (x >= y) {
            // Draw 8 octants with thickness
            for (int t = -halfThick; t <= halfThick; ++t) {
                setPixel(img, w, h, channels, cx + x, cy + y + t, color);
                setPixel(img, w, h, channels, cx + y, cy + x + t, color);
                setPixel(img, w, h, channels, cx - y, cy + x + t, color);
                setPixel(img, w, h, channels, cx - x, cy + y + t, color);
                setPixel(img, w, h, channels, cx - x, cy - y + t, color);
                setPixel(img, w, h, channels, cx - y, cy - x + t, color);
                setPixel(img, w, h, channels, cx + y, cy - x + t, color);
                setPixel(img, w, h, channels, cx + x, cy - y + t, color);
            }
            
            y++;
            if (err <= 0) {
                err += 2 * y + 1;
            }
            if (err > 0) {
                x--;
                err -= 2 * x + 1;
            }
        }
    }
}

/**
 * @brief Draw an ellipse
 */
inline void drawEllipse(uint8_t* img, int w, int h, int channels,
                       int cx, int cy, int a, int b,
                       const Color& color, int thickness = 1, bool filled = false) {
    if (filled) {
        for (int y = -b; y <= b; ++y) {
            for (int x = -a; x <= a; ++x) {
                float ellipse = (static_cast<float>(x * x) / (a * a)) + 
                               (static_cast<float>(y * y) / (b * b));
                if (ellipse <= 1.0f) {
                    setPixel(img, w, h, channels, cx + x, cy + y, color);
                }
            }
        }
    } else {
        // Parametric drawing
        float step = 1.0f / std::max(a, b);
        float prevX = cx + a;
        float prevY = cy;
        
        for (float t = step; t <= 2 * 3.14159265f + step; t += step) {
            float x = cx + a * std::cos(t);
            float y = cy + b * std::sin(t);
            drawLine(img, w, h, channels, 
                    static_cast<int>(prevX), static_cast<int>(prevY),
                    static_cast<int>(x), static_cast<int>(y), color, thickness);
            prevX = x;
            prevY = y;
        }
    }
}

/**
 * @brief Draw a polygon
 */
inline void drawPolygon(uint8_t* img, int w, int h, int channels,
                       const std::vector<Point2i>& points,
                       const Color& color, int thickness = 1, bool closed = true) {
    if (points.size() < 2) return;
    
    for (size_t i = 0; i < points.size() - 1; ++i) {
        drawLine(img, w, h, channels, 
                points[i].x, points[i].y, 
                points[i + 1].x, points[i + 1].y, color, thickness);
    }
    
    if (closed && points.size() >= 3) {
        drawLine(img, w, h, channels,
                points.back().x, points.back().y,
                points.front().x, points.front().y, color, thickness);
    }
}

/**
 * @brief Fill a convex polygon
 */
inline void fillConvexPoly(uint8_t* img, int w, int h, int channels,
                          const std::vector<Point2i>& points,
                          const Color& color) {
    if (points.size() < 3) return;
    
    // Find bounding box
    int minY = points[0].y, maxY = points[0].y;
    for (const auto& p : points) {
        minY = std::min(minY, p.y);
        maxY = std::max(maxY, p.y);
    }
    
    minY = std::max(0, minY);
    maxY = std::min(h - 1, maxY);
    
    // Scanline fill
    for (int y = minY; y <= maxY; ++y) {
        std::vector<int> intersections;
        
        for (size_t i = 0; i < points.size(); ++i) {
            size_t j = (i + 1) % points.size();
            int y1 = points[i].y, y2 = points[j].y;
            int x1 = points[i].x, x2 = points[j].x;
            
            if ((y1 <= y && y < y2) || (y2 <= y && y < y1)) {
                float t = static_cast<float>(y - y1) / (y2 - y1);
                int x = static_cast<int>(x1 + t * (x2 - x1));
                intersections.push_back(x);
            }
        }
        
        std::sort(intersections.begin(), intersections.end());
        
        for (size_t i = 0; i + 1 < intersections.size(); i += 2) {
            int xStart = std::max(0, intersections[i]);
            int xEnd = std::min(w - 1, intersections[i + 1]);
            for (int x = xStart; x <= xEnd; ++x) {
                setPixel(img, w, h, channels, x, y, color);
            }
        }
    }
}

// ============================================================================
// Keypoint Visualization
// ============================================================================

/**
 * @brief Draw keypoints on image
 */
inline void drawKeypoints(uint8_t* img, int w, int h, int channels,
                         const std::vector<DrawKeypoint>& keypoints,
                         const Color& color = Color::Green(),
                         int flags = 0) {
    // flags: 0 = draw points, 1 = draw rich (size + angle)
    
    for (const auto& kp : keypoints) {
        int x = static_cast<int>(kp.x);
        int y = static_cast<int>(kp.y);
        int radius = std::max(1, static_cast<int>(kp.size / 2));
        
        if (flags == 0) {
            // Simple point
            drawCircle(img, w, h, channels, x, y, radius, color, 1, true);
        } else {
            // Rich keypoint
            drawCircle(img, w, h, channels, x, y, radius, color, 1, false);
            
            // Draw orientation line if available
            if (kp.angle >= 0) {
                float rad = kp.angle * 3.14159265f / 180.0f;
                int x2 = x + static_cast<int>(radius * std::cos(rad));
                int y2 = y + static_cast<int>(radius * std::sin(rad));
                drawLine(img, w, h, channels, x, y, x2, y2, color, 1);
            }
        }
    }
}

// ============================================================================
// Match Visualization
// ============================================================================

/**
 * @brief Create side-by-side image and draw matches
 */
inline std::vector<uint8_t> drawMatches(
    const uint8_t* img1, int w1, int h1,
    const std::vector<DrawKeypoint>& kp1,
    const uint8_t* img2, int w2, int h2,
    const std::vector<DrawKeypoint>& kp2,
    const std::vector<DrawMatch>& matches,
    int channels,
    int& outW, int& outH,
    const Color& matchColor = Color::Green(),
    const Color& keypointColor = Color::Blue()) {
    
    // Create output image
    outW = w1 + w2;
    outH = std::max(h1, h2);
    std::vector<uint8_t> out(outW * outH * channels, 0);
    
    // Copy image 1
    for (int y = 0; y < h1; ++y) {
        for (int x = 0; x < w1; ++x) {
            for (int c = 0; c < channels; ++c) {
                out[(y * outW + x) * channels + c] = img1[(y * w1 + x) * channels + c];
            }
        }
    }
    
    // Copy image 2
    for (int y = 0; y < h2; ++y) {
        for (int x = 0; x < w2; ++x) {
            for (int c = 0; c < channels; ++c) {
                out[(y * outW + (x + w1)) * channels + c] = img2[(y * w2 + x) * channels + c];
            }
        }
    }
    
    // Draw matches
    for (const auto& m : matches) {
        if (m.queryIdx < 0 || static_cast<size_t>(m.queryIdx) >= kp1.size()) continue;
        if (m.trainIdx < 0 || static_cast<size_t>(m.trainIdx) >= kp2.size()) continue;
        
        int x1 = static_cast<int>(kp1[m.queryIdx].x);
        int y1 = static_cast<int>(kp1[m.queryIdx].y);
        int x2 = static_cast<int>(kp2[m.trainIdx].x) + w1;
        int y2 = static_cast<int>(kp2[m.trainIdx].y);
        
        drawLine(out.data(), outW, outH, channels, x1, y1, x2, y2, matchColor, 1);
        drawCircle(out.data(), outW, outH, channels, x1, y1, 3, keypointColor, 1, true);
        drawCircle(out.data(), outW, outH, channels, x2, y2, 3, keypointColor, 1, true);
    }
    
    return out;
}

// ============================================================================
// Text Rendering (Simple)
// ============================================================================

// Simple 5x7 font
namespace font {

inline const uint8_t CHAR_WIDTH = 5;
inline const uint8_t CHAR_HEIGHT = 7;

// Simplified glyph data for digits and some characters
inline std::array<uint8_t, 7> getGlyph(char c) {
    static const std::array<std::array<uint8_t, 7>, 128> glyphs = []() {
        std::array<std::array<uint8_t, 7>, 128> g{};
        // Initialize all to empty
        for (auto& ch : g) ch = {0, 0, 0, 0, 0, 0, 0};
        
        // Digit 0
        g['0'] = {0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110};
        g['1'] = {0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110};
        g['2'] = {0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111};
        g['3'] = {0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110};
        g['4'] = {0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010};
        g['5'] = {0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110};
        g['6'] = {0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110};
        g['7'] = {0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000};
        g['8'] = {0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110};
        g['9'] = {0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100};
        
        // Letters (uppercase)
        g['A'] = {0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001};
        g['B'] = {0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110};
        g['C'] = {0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110};
        g['D'] = {0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110};
        g['E'] = {0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111};
        g['F'] = {0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000};
        g['G'] = {0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111};
        g['H'] = {0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001};
        g['I'] = {0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110};
        g['J'] = {0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100};
        g['K'] = {0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001};
        g['L'] = {0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111};
        g['M'] = {0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001};
        g['N'] = {0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001};
        g['O'] = {0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110};
        g['P'] = {0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000};
        g['Q'] = {0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101};
        g['R'] = {0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001};
        g['S'] = {0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110};
        g['T'] = {0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100};
        g['U'] = {0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110};
        g['V'] = {0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100};
        g['W'] = {0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001};
        g['X'] = {0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001};
        g['Y'] = {0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100};
        g['Z'] = {0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111};
        
        // Symbols
        g['.'] = {0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100};
        g[','] = {0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00100, 0b01000};
        g[':'] = {0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000};
        g['-'] = {0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000};
        g['+'] = {0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000};
        g['('] = {0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010};
        g[')'] = {0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000};
        g[' '] = {0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000};
        
        return g;
    }();
    
    int idx = static_cast<unsigned char>(c);
    if (idx >= 128) return {0, 0, 0, 0, 0, 0, 0};
    return glyphs[idx];
}

} // namespace font

/**
 * @brief Draw text on image
 */
inline void putText(uint8_t* img, int w, int h, int channels,
                   const std::string& text,
                   int x, int y,
                   const Color& color,
                   int scale = 1) {
    int curX = x;
    
    for (char c : text) {
        auto glyph = font::getGlyph(std::toupper(c));
        
        for (int gy = 0; gy < font::CHAR_HEIGHT; ++gy) {
            for (int gx = 0; gx < font::CHAR_WIDTH; ++gx) {
                if (glyph[gy] & (1 << (font::CHAR_WIDTH - 1 - gx))) {
                    // Draw scaled pixel
                    for (int sy = 0; sy < scale; ++sy) {
                        for (int sx = 0; sx < scale; ++sx) {
                            setPixel(img, w, h, channels,
                                    curX + gx * scale + sx,
                                    y + gy * scale + sy, color);
                        }
                    }
                }
            }
        }
        
        curX += (font::CHAR_WIDTH + 1) * scale;
    }
}

// ============================================================================
// Annotation Utilities
// ============================================================================

/**
 * @brief Draw bounding box with label
 */
inline void drawBoundingBox(uint8_t* img, int w, int h, int channels,
                           int x1, int y1, int x2, int y2,
                           const std::string& label,
                           const Color& boxColor = Color::Green(),
                           const Color& textColor = Color::White(),
                           int thickness = 2) {
    drawRectangle(img, w, h, channels, x1, y1, x2, y2, boxColor, thickness);
    
    // Draw label background
    int textWidth = static_cast<int>(label.size()) * (font::CHAR_WIDTH + 1);
    int textHeight = font::CHAR_HEIGHT;
    
    int labelY = (y1 >= textHeight + 2) ? y1 - textHeight - 2 : y2 + 2;
    
    drawRectangle(img, w, h, channels, 
                 x1, labelY, x1 + textWidth + 2, labelY + textHeight + 2,
                 boxColor, 1, true);
    
    putText(img, w, h, channels, label, x1 + 1, labelY + 1, textColor, 1);
}

/**
 * @brief Draw crosshair
 */
inline void drawCrosshair(uint8_t* img, int w, int h, int channels,
                         int cx, int cy, int size,
                         const Color& color, int thickness = 1) {
    drawLine(img, w, h, channels, cx - size, cy, cx + size, cy, color, thickness);
    drawLine(img, w, h, channels, cx, cy - size, cx, cy + size, color, thickness);
}

/**
 * @brief Draw arrow
 */
inline void drawArrow(uint8_t* img, int w, int h, int channels,
                     int x1, int y1, int x2, int y2,
                     const Color& color, int thickness = 1, int tipLength = 10) {
    drawLine(img, w, h, channels, x1, y1, x2, y2, color, thickness);
    
    // Draw arrowhead
    float angle = std::atan2(static_cast<float>(y2 - y1), static_cast<float>(x2 - x1));
    float tipAngle1 = angle + 2.5f;  // ~143 degrees
    float tipAngle2 = angle - 2.5f;
    
    int tipX1 = x2 - static_cast<int>(tipLength * std::cos(tipAngle1));
    int tipY1 = y2 - static_cast<int>(tipLength * std::sin(tipAngle1));
    int tipX2 = x2 - static_cast<int>(tipLength * std::cos(tipAngle2));
    int tipY2 = y2 - static_cast<int>(tipLength * std::sin(tipAngle2));
    
    drawLine(img, w, h, channels, x2, y2, tipX1, tipY1, color, thickness);
    drawLine(img, w, h, channels, x2, y2, tipX2, tipY2, color, thickness);
}

// ============================================================================
// Random Color Generation
// ============================================================================

/**
 * @brief Generate random distinct colors
 */
inline std::vector<Color> generateColors(size_t count, unsigned int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(50, 255);
    
    std::vector<Color> colors;
    colors.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        // Use golden ratio for hue distribution
        float hue = std::fmod(i * 0.618033988749895f, 1.0f) * 360.0f;
        float saturation = 0.7f + 0.3f * (dist(rng) / 255.0f);
        float value = 0.7f + 0.3f * (dist(rng) / 255.0f);
        
        // HSV to RGB conversion
        float c = value * saturation;
        float x = c * (1 - std::abs(std::fmod(hue / 60.0f, 2.0f) - 1));
        float m = value - c;
        
        float r, g, b;
        if (hue < 60) { r = c; g = x; b = 0; }
        else if (hue < 120) { r = x; g = c; b = 0; }
        else if (hue < 180) { r = 0; g = c; b = x; }
        else if (hue < 240) { r = 0; g = x; b = c; }
        else if (hue < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }
        
        colors.emplace_back(
            static_cast<uint8_t>((r + m) * 255),
            static_cast<uint8_t>((g + m) * 255),
            static_cast<uint8_t>((b + m) * 255)
        );
    }
    
    return colors;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Clamp value to range
 */
template<typename T>
inline T clamp(T value, T minVal, T maxVal) {
    return std::max(minVal, std::min(maxVal, value));
}

/**
 * @brief Linear interpolation
 */
template<typename T>
inline T lerp(T a, T b, float t) {
    return static_cast<T>(a + t * (b - a));
}

/**
 * @brief Map value from one range to another
 */
inline float mapRange(float value, float inMin, float inMax, float outMin, float outMax) {
    return outMin + (value - inMin) * (outMax - outMin) / (inMax - inMin);
}

/**
 * @brief Compute Euclidean distance
 */
inline float distance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

/**
 * @brief Compute squared distance
 */
inline float distanceSquared(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return dx * dx + dy * dy;
}

/**
 * @brief Convert degrees to radians
 */
inline float degToRad(float degrees) {
    return degrees * 3.14159265358979f / 180.0f;
}

/**
 * @brief Convert radians to degrees
 */
inline float radToDeg(float radians) {
    return radians * 180.0f / 3.14159265358979f;
}

} // namespace utils
} // namespace neurova

#endif // NEUROVA_UTILS_HPP
