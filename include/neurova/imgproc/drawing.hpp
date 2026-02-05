// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file drawing.hpp
 * @brief Image drawing functions
 */

#ifndef NEUROVA_IMGPROC_DRAWING_HPP
#define NEUROVA_IMGPROC_DRAWING_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

namespace neurova {
namespace imgproc {

// Line types
constexpr int LINE_4 = 4;
constexpr int LINE_8 = 8;
constexpr int LINE_AA = 16;

// Filled shapes
constexpr int FILLED = -1;

/**
 * @brief Color structure
 */
struct Scalar {
    float val[4];
    
    Scalar() : val{0, 0, 0, 0} {}
    Scalar(float v0) : val{v0, 0, 0, 0} {}
    Scalar(float v0, float v1, float v2) : val{v0, v1, v2, 0} {}
    Scalar(float v0, float v1, float v2, float v3) : val{v0, v1, v2, v3} {}
    
    float& operator[](int i) { return val[i]; }
    const float& operator[](int i) const { return val[i]; }
};

/**
 * @brief Point structure
 */
struct Point {
    int x, y;
    Point() : x(0), y(0) {}
    Point(int x_, int y_) : x(x_), y(y_) {}
};

struct Point2f {
    float x, y;
    Point2f() : x(0), y(0) {}
    Point2f(float x_, float y_) : x(x_), y(y_) {}
};

/**
 * @brief Set pixel with bounds checking
 */
inline void setPixel(
    float* image, int width, int height, int channels,
    int x, int y, const Scalar& color
) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    
    int idx = (y * width + x) * channels;
    for (int c = 0; c < channels; ++c) {
        image[idx + c] = color[c];
    }
}

/**
 * @brief Set pixel with alpha blending
 */
inline void setPixelAlpha(
    float* image, int width, int height, int channels,
    int x, int y, const Scalar& color, float alpha
) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    
    int idx = (y * width + x) * channels;
    for (int c = 0; c < channels; ++c) {
        image[idx + c] = image[idx + c] * (1 - alpha) + color[c] * alpha;
    }
}

/**
 * @brief Draw a line using Bresenham's algorithm
 */
inline void line(
    float* image, int width, int height, int channels,
    Point pt1, Point pt2,
    const Scalar& color,
    int thickness = 1,
    int lineType = LINE_8
) {
    int x0 = pt1.x, y0 = pt1.y;
    int x1 = pt2.x, y1 = pt2.y;
    
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    
    while (true) {
        // Draw thick line
        if (thickness == 1) {
            setPixel(image, width, height, channels, x0, y0, color);
        } else {
            int r = thickness / 2;
            for (int ty = -r; ty <= r; ++ty) {
                for (int tx = -r; tx <= r; ++tx) {
                    if (tx * tx + ty * ty <= r * r) {
                        setPixel(image, width, height, channels, x0 + tx, y0 + ty, color);
                    }
                }
            }
        }
        
        if (x0 == x1 && y0 == y1) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

/**
 * @brief Draw anti-aliased line (Wu's algorithm)
 */
inline void lineAA(
    float* image, int width, int height, int channels,
    Point pt1, Point pt2,
    const Scalar& color
) {
    float x0 = static_cast<float>(pt1.x);
    float y0 = static_cast<float>(pt1.y);
    float x1 = static_cast<float>(pt2.x);
    float y1 = static_cast<float>(pt2.y);
    
    bool steep = std::abs(y1 - y0) > std::abs(x1 - x0);
    
    if (steep) {
        std::swap(x0, y0);
        std::swap(x1, y1);
    }
    if (x0 > x1) {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }
    
    float dx = x1 - x0;
    float dy = y1 - y0;
    float gradient = (dx == 0) ? 1.0f : dy / dx;
    
    // First endpoint
    float xend = std::round(x0);
    float yend = y0 + gradient * (xend - x0);
    float xgap = 1 - (x0 + 0.5f - std::floor(x0 + 0.5f));
    int xpxl1 = static_cast<int>(xend);
    int ypxl1 = static_cast<int>(std::floor(yend));
    
    if (steep) {
        setPixelAlpha(image, width, height, channels, ypxl1, xpxl1, color, 
                     (1 - (yend - std::floor(yend))) * xgap);
        setPixelAlpha(image, width, height, channels, ypxl1 + 1, xpxl1, color,
                     (yend - std::floor(yend)) * xgap);
    } else {
        setPixelAlpha(image, width, height, channels, xpxl1, ypxl1, color,
                     (1 - (yend - std::floor(yend))) * xgap);
        setPixelAlpha(image, width, height, channels, xpxl1, ypxl1 + 1, color,
                     (yend - std::floor(yend)) * xgap);
    }
    
    float intery = yend + gradient;
    
    // Second endpoint
    xend = std::round(x1);
    yend = y1 + gradient * (xend - x1);
    xgap = x1 + 0.5f - std::floor(x1 + 0.5f);
    int xpxl2 = static_cast<int>(xend);
    int ypxl2 = static_cast<int>(std::floor(yend));
    
    if (steep) {
        setPixelAlpha(image, width, height, channels, ypxl2, xpxl2, color,
                     (1 - (yend - std::floor(yend))) * xgap);
        setPixelAlpha(image, width, height, channels, ypxl2 + 1, xpxl2, color,
                     (yend - std::floor(yend)) * xgap);
    } else {
        setPixelAlpha(image, width, height, channels, xpxl2, ypxl2, color,
                     (1 - (yend - std::floor(yend))) * xgap);
        setPixelAlpha(image, width, height, channels, xpxl2, ypxl2 + 1, color,
                     (yend - std::floor(yend)) * xgap);
    }
    
    // Main loop
    for (int x = xpxl1 + 1; x < xpxl2; ++x) {
        if (steep) {
            setPixelAlpha(image, width, height, channels, 
                         static_cast<int>(std::floor(intery)), x, color,
                         1 - (intery - std::floor(intery)));
            setPixelAlpha(image, width, height, channels,
                         static_cast<int>(std::floor(intery)) + 1, x, color,
                         intery - std::floor(intery));
        } else {
            setPixelAlpha(image, width, height, channels,
                         x, static_cast<int>(std::floor(intery)), color,
                         1 - (intery - std::floor(intery)));
            setPixelAlpha(image, width, height, channels,
                         x, static_cast<int>(std::floor(intery)) + 1, color,
                         intery - std::floor(intery));
        }
        intery += gradient;
    }
}

/**
 * @brief Draw a rectangle
 */
inline void rectangle(
    float* image, int width, int height, int channels,
    Point pt1, Point pt2,
    const Scalar& color,
    int thickness = 1,
    int lineType = LINE_8
) {
    int x1 = std::min(pt1.x, pt2.x);
    int y1 = std::min(pt1.y, pt2.y);
    int x2 = std::max(pt1.x, pt2.x);
    int y2 = std::max(pt1.y, pt2.y);
    
    if (thickness == FILLED) {
        for (int y = y1; y <= y2; ++y) {
            for (int x = x1; x <= x2; ++x) {
                setPixel(image, width, height, channels, x, y, color);
            }
        }
    } else {
        line(image, width, height, channels, {x1, y1}, {x2, y1}, color, thickness, lineType);
        line(image, width, height, channels, {x2, y1}, {x2, y2}, color, thickness, lineType);
        line(image, width, height, channels, {x2, y2}, {x1, y2}, color, thickness, lineType);
        line(image, width, height, channels, {x1, y2}, {x1, y1}, color, thickness, lineType);
    }
}

/**
 * @brief Draw a circle using Midpoint algorithm
 */
inline void circle(
    float* image, int width, int height, int channels,
    Point center, int radius,
    const Scalar& color,
    int thickness = 1,
    int lineType = LINE_8
) {
    int cx = center.x;
    int cy = center.y;
    
    if (thickness == FILLED) {
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                if (x * x + y * y <= radius * radius) {
                    setPixel(image, width, height, channels, cx + x, cy + y, color);
                }
            }
        }
        return;
    }
    
    // Midpoint circle algorithm
    int x = radius;
    int y = 0;
    int err = 0;
    
    while (x >= y) {
        for (int t = 0; t < thickness; ++t) {
            int r = radius - t;
            setPixel(image, width, height, channels, cx + x, cy + y, color);
            setPixel(image, width, height, channels, cx + y, cy + x, color);
            setPixel(image, width, height, channels, cx - y, cy + x, color);
            setPixel(image, width, height, channels, cx - x, cy + y, color);
            setPixel(image, width, height, channels, cx - x, cy - y, color);
            setPixel(image, width, height, channels, cx - y, cy - x, color);
            setPixel(image, width, height, channels, cx + y, cy - x, color);
            setPixel(image, width, height, channels, cx + x, cy - y, color);
        }
        
        if (err <= 0) {
            ++y;
            err += 2 * y + 1;
        }
        if (err > 0) {
            --x;
            err -= 2 * x + 1;
        }
    }
}

/**
 * @brief Draw an ellipse
 */
inline void ellipse(
    float* image, int width, int height, int channels,
    Point center, int axisX, int axisY,
    float angle, float startAngle, float endAngle,
    const Scalar& color,
    int thickness = 1,
    int lineType = LINE_8
) {
    float radAngle = angle * 3.14159265f / 180.0f;
    float cosA = std::cos(radAngle);
    float sinA = std::sin(radAngle);
    
    float startRad = startAngle * 3.14159265f / 180.0f;
    float endRad = endAngle * 3.14159265f / 180.0f;
    
    int nPoints = std::max(axisX, axisY) * 4;
    float step = (endRad - startRad) / nPoints;
    
    Point prevPt;
    bool first = true;
    
    for (int i = 0; i <= nPoints; ++i) {
        float t = startRad + i * step;
        float x = axisX * std::cos(t);
        float y = axisY * std::sin(t);
        
        // Rotate
        float rx = x * cosA - y * sinA + center.x;
        float ry = x * sinA + y * cosA + center.y;
        
        Point pt(static_cast<int>(rx), static_cast<int>(ry));
        
        if (!first) {
            line(image, width, height, channels, prevPt, pt, color, thickness, lineType);
        }
        
        prevPt = pt;
        first = false;
    }
}

/**
 * @brief Draw polylines
 */
inline void polylines(
    float* image, int width, int height, int channels,
    const std::vector<Point>& pts,
    bool isClosed,
    const Scalar& color,
    int thickness = 1,
    int lineType = LINE_8
) {
    if (pts.size() < 2) return;
    
    for (size_t i = 0; i < pts.size() - 1; ++i) {
        line(image, width, height, channels, pts[i], pts[i + 1], color, thickness, lineType);
    }
    
    if (isClosed && pts.size() > 2) {
        line(image, width, height, channels, pts.back(), pts.front(), color, thickness, lineType);
    }
}

/**
 * @brief Fill a convex polygon
 */
inline void fillConvexPoly(
    float* image, int width, int height, int channels,
    const std::vector<Point>& pts,
    const Scalar& color
) {
    if (pts.size() < 3) return;
    
    // Find bounding box
    int minX = pts[0].x, maxX = pts[0].x;
    int minY = pts[0].y, maxY = pts[0].y;
    
    for (const auto& pt : pts) {
        minX = std::min(minX, pt.x);
        maxX = std::max(maxX, pt.x);
        minY = std::min(minY, pt.y);
        maxY = std::max(maxY, pt.y);
    }
    
    minX = std::max(0, minX);
    maxX = std::min(width - 1, maxX);
    minY = std::max(0, minY);
    maxY = std::min(height - 1, maxY);
    
    // Scanline fill
    for (int y = minY; y <= maxY; ++y) {
        std::vector<int> intersections;
        
        for (size_t i = 0; i < pts.size(); ++i) {
            size_t j = (i + 1) % pts.size();
            int y1 = pts[i].y, y2 = pts[j].y;
            int x1 = pts[i].x, x2 = pts[j].x;
            
            if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
                float t = static_cast<float>(y - y1) / (y2 - y1);
                int x = static_cast<int>(x1 + t * (x2 - x1));
                intersections.push_back(x);
            }
        }
        
        std::sort(intersections.begin(), intersections.end());
        
        for (size_t i = 0; i + 1 < intersections.size(); i += 2) {
            for (int x = std::max(0, intersections[i]); 
                 x <= std::min(width - 1, intersections[i + 1]); ++x) {
                setPixel(image, width, height, channels, x, y, color);
            }
        }
    }
}

/**
 * @brief Draw text (basic bitmap font)
 */
inline void putText(
    float* image, int width, int height, int channels,
    const std::string& text,
    Point org,
    int fontFace,
    float fontScale,
    const Scalar& color,
    int thickness = 1
) {
    // Simple 5x7 bitmap font for ASCII printable characters
    // This is a minimal implementation
    
    int charWidth = static_cast<int>(6 * fontScale);
    int charHeight = static_cast<int>(8 * fontScale);
    
    int x = org.x;
    int y = org.y;
    
    for (char c : text) {
        if (c < 32 || c > 126) continue;
        
        // Draw a simple placeholder rectangle for each character
        // In production, use a proper font renderer
        rectangle(image, width, height, channels,
                 {x + 1, y - charHeight + 2},
                 {x + charWidth - 2, y - 2},
                 color, thickness);
        
        x += charWidth;
    }
}

/**
 * @brief Draw an arrow
 */
inline void arrowedLine(
    float* image, int width, int height, int channels,
    Point pt1, Point pt2,
    const Scalar& color,
    int thickness = 1,
    int lineType = LINE_8,
    float tipLength = 0.1f
) {
    line(image, width, height, channels, pt1, pt2, color, thickness, lineType);
    
    float dx = static_cast<float>(pt2.x - pt1.x);
    float dy = static_cast<float>(pt2.y - pt1.y);
    float length = std::sqrt(dx * dx + dy * dy);
    
    if (length < 1e-6f) return;
    
    float angle = std::atan2(dy, dx);
    float tipLen = length * tipLength;
    float arrowAngle = 0.5f;  // 30 degrees
    
    Point tip1(
        static_cast<int>(pt2.x - tipLen * std::cos(angle - arrowAngle)),
        static_cast<int>(pt2.y - tipLen * std::sin(angle - arrowAngle))
    );
    Point tip2(
        static_cast<int>(pt2.x - tipLen * std::cos(angle + arrowAngle)),
        static_cast<int>(pt2.y - tipLen * std::sin(angle + arrowAngle))
    );
    
    line(image, width, height, channels, pt2, tip1, color, thickness, lineType);
    line(image, width, height, channels, pt2, tip2, color, thickness, lineType);
}

/**
 * @brief Draw a marker
 */
inline void drawMarker(
    float* image, int width, int height, int channels,
    Point position,
    const Scalar& color,
    int markerType = 0,  // 0=cross, 1=tilted cross, 2=star, 3=diamond, 4=square, 5=triangle up, 6=triangle down
    int markerSize = 20,
    int thickness = 1,
    int lineType = LINE_8
) {
    int s = markerSize / 2;
    
    switch (markerType) {
        case 0:  // Cross
            line(image, width, height, channels,
                 {position.x - s, position.y}, {position.x + s, position.y},
                 color, thickness, lineType);
            line(image, width, height, channels,
                 {position.x, position.y - s}, {position.x, position.y + s},
                 color, thickness, lineType);
            break;
            
        case 1:  // Tilted cross
            line(image, width, height, channels,
                 {position.x - s, position.y - s}, {position.x + s, position.y + s},
                 color, thickness, lineType);
            line(image, width, height, channels,
                 {position.x + s, position.y - s}, {position.x - s, position.y + s},
                 color, thickness, lineType);
            break;
            
        case 2:  // Star (cross + tilted)
            drawMarker(image, width, height, channels, position, color, 0, markerSize, thickness, lineType);
            drawMarker(image, width, height, channels, position, color, 1, markerSize, thickness, lineType);
            break;
            
        case 3:  // Diamond
            polylines(image, width, height, channels, {
                {position.x, position.y - s},
                {position.x + s, position.y},
                {position.x, position.y + s},
                {position.x - s, position.y}
            }, true, color, thickness, lineType);
            break;
            
        case 4:  // Square
            rectangle(image, width, height, channels,
                     {position.x - s, position.y - s},
                     {position.x + s, position.y + s},
                     color, thickness, lineType);
            break;
            
        case 5:  // Triangle up
            polylines(image, width, height, channels, {
                {position.x, position.y - s},
                {position.x + s, position.y + s},
                {position.x - s, position.y + s}
            }, true, color, thickness, lineType);
            break;
            
        case 6:  // Triangle down
            polylines(image, width, height, channels, {
                {position.x, position.y + s},
                {position.x + s, position.y - s},
                {position.x - s, position.y - s}
            }, true, color, thickness, lineType);
            break;
    }
}

} // namespace imgproc
} // namespace neurova

#endif // NEUROVA_IMGPROC_DRAWING_HPP
