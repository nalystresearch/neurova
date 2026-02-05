// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file contours.hpp
 * @brief Contour detection and analysis
 */

#ifndef NEUROVA_IMGPROC_CONTOURS_HPP
#define NEUROVA_IMGPROC_CONTOURS_HPP

#include "drawing.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>

namespace neurova {
namespace imgproc {

// Contour retrieval modes
constexpr int RETR_EXTERNAL = 0;
constexpr int RETR_LIST = 1;
constexpr int RETR_CCOMP = 2;
constexpr int RETR_TREE = 3;

// Contour approximation methods
constexpr int CHAIN_APPROX_NONE = 0;
constexpr int CHAIN_APPROX_SIMPLE = 1;
constexpr int CHAIN_APPROX_TC89_L1 = 2;
constexpr int CHAIN_APPROX_TC89_KCOS = 3;

/**
 * @brief Rectangle structure
 */
struct Rect {
    int x, y, width, height;
    
    Rect() : x(0), y(0), width(0), height(0) {}
    Rect(int x_, int y_, int w_, int h_) : x(x_), y(y_), width(w_), height(h_) {}
    
    int area() const { return width * height; }
    Point tl() const { return {x, y}; }
    Point br() const { return {x + width, y + height}; }
};

/**
 * @brief Rotated rectangle
 */
struct RotatedRect {
    Point2f center;
    float width, height;
    float angle;
    
    RotatedRect() : width(0), height(0), angle(0) {}
    RotatedRect(Point2f c, float w, float h, float a) 
        : center(c), width(w), height(h), angle(a) {}
};

/**
 * @brief Image moments
 */
struct Moments {
    double m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
    double mu20, mu11, mu02, mu30, mu21, mu12, mu03;
    double nu20, nu11, nu02, nu30, nu21, nu12, nu03;
    
    Moments() : m00(0), m10(0), m01(0), m20(0), m11(0), m02(0),
                m30(0), m21(0), m12(0), m03(0),
                mu20(0), mu11(0), mu02(0), mu30(0), mu21(0), mu12(0), mu03(0),
                nu20(0), nu11(0), nu02(0), nu30(0), nu21(0), nu12(0), nu03(0) {}
};

/**
 * @brief Find contours in binary image
 */
inline std::vector<std::vector<Point>> findContours(
    const float* binary, int width, int height,
    int mode = RETR_LIST,
    int method = CHAIN_APPROX_SIMPLE
) {
    std::vector<std::vector<Point>> contours;
    std::vector<bool> visited(width * height, false);
    
    // Copy image to work with
    std::vector<uint8_t> image(width * height);
    for (int i = 0; i < width * height; ++i) {
        image[i] = (binary[i] > 127) ? 255 : 0;
    }
    
    // Direction vectors for 8-connectivity
    const int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            
            // Look for boundary pixel (white pixel with at least one black neighbor)
            if (image[idx] == 0 || visited[idx]) continue;
            
            bool isBoundary = false;
            for (int d = 0; d < 8; ++d) {
                int nx = x + dx[d];
                int ny = y + dy[d];
                if (nx < 0 || nx >= width || ny < 0 || ny >= height ||
                    image[ny * width + nx] == 0) {
                    isBoundary = true;
                    break;
                }
            }
            
            if (!isBoundary) continue;
            
            // Trace contour using Moore-Neighbor tracing
            std::vector<Point> contour;
            int startX = x, startY = y;
            int cx = x, cy = y;
            int dir = 0;  // Start direction
            
            do {
                if (method == CHAIN_APPROX_NONE || contour.empty() ||
                    contour.back().x != cx || contour.back().y != cy) {
                    contour.push_back({cx, cy});
                }
                visited[cy * width + cx] = true;
                
                // Find next boundary pixel
                bool found = false;
                for (int i = 0; i < 8; ++i) {
                    int newDir = (dir + 8 - 1 + i) % 8;  // Start from previous direction
                    int nx = cx + dx[newDir];
                    int ny = cy + dy[newDir];
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                        image[ny * width + nx] > 0) {
                        cx = nx;
                        cy = ny;
                        dir = newDir;
                        found = true;
                        break;
                    }
                }
                
                if (!found) break;
                
            } while (cx != startX || cy != startY);
            
            // Simplify contour if using CHAIN_APPROX_SIMPLE
            if (method == CHAIN_APPROX_SIMPLE && contour.size() > 4) {
                std::vector<Point> simplified;
                simplified.push_back(contour[0]);
                
                for (size_t i = 1; i < contour.size() - 1; ++i) {
                    int dx1 = contour[i].x - contour[i - 1].x;
                    int dy1 = contour[i].y - contour[i - 1].y;
                    int dx2 = contour[i + 1].x - contour[i].x;
                    int dy2 = contour[i + 1].y - contour[i].y;
                    
                    // Keep point if direction changes
                    if (dx1 != dx2 || dy1 != dy2) {
                        simplified.push_back(contour[i]);
                    }
                }
                simplified.push_back(contour.back());
                contour = simplified;
            }
            
            if (contour.size() >= 3) {
                contours.push_back(contour);
            }
        }
    }
    
    return contours;
}

/**
 * @brief Draw contours on image
 */
inline void drawContours(
    float* image, int width, int height, int channels,
    const std::vector<std::vector<Point>>& contours,
    int contourIdx,
    const Scalar& color,
    int thickness = 1,
    int lineType = LINE_8
) {
    auto draw = [&](const std::vector<Point>& contour) {
        if (thickness == FILLED) {
            fillConvexPoly(image, width, height, channels, contour, color);
        } else {
            polylines(image, width, height, channels, contour, true, color, thickness, lineType);
        }
    };
    
    if (contourIdx < 0) {
        for (const auto& contour : contours) {
            draw(contour);
        }
    } else if (contourIdx < static_cast<int>(contours.size())) {
        draw(contours[contourIdx]);
    }
}

/**
 * @brief Calculate contour area
 */
inline double contourArea(const std::vector<Point>& contour, bool oriented = false) {
    if (contour.size() < 3) return 0.0;
    
    double area = 0.0;
    int n = static_cast<int>(contour.size());
    
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += static_cast<double>(contour[i].x) * contour[j].y;
        area -= static_cast<double>(contour[j].x) * contour[i].y;
    }
    
    area /= 2.0;
    return oriented ? area : std::abs(area);
}

/**
 * @brief Calculate arc length (perimeter)
 */
inline double arcLength(const std::vector<Point>& curve, bool closed = false) {
    double length = 0.0;
    int n = static_cast<int>(curve.size());
    
    for (int i = 0; i < n - 1; ++i) {
        double dx = curve[i + 1].x - curve[i].x;
        double dy = curve[i + 1].y - curve[i].y;
        length += std::sqrt(dx * dx + dy * dy);
    }
    
    if (closed && n > 2) {
        double dx = curve[0].x - curve[n - 1].x;
        double dy = curve[0].y - curve[n - 1].y;
        length += std::sqrt(dx * dx + dy * dy);
    }
    
    return length;
}

/**
 * @brief Get bounding rectangle
 */
inline Rect boundingRect(const std::vector<Point>& points) {
    if (points.empty()) return Rect();
    
    int minX = points[0].x, maxX = points[0].x;
    int minY = points[0].y, maxY = points[0].y;
    
    for (const auto& pt : points) {
        minX = std::min(minX, pt.x);
        maxX = std::max(maxX, pt.x);
        minY = std::min(minY, pt.y);
        maxY = std::max(maxY, pt.y);
    }
    
    return Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
}

/**
 * @brief Get minimum area bounding rectangle
 */
inline RotatedRect minAreaRect(const std::vector<Point>& points) {
    if (points.size() < 3) {
        auto rect = boundingRect(points);
        return RotatedRect(
            Point2f(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f),
            static_cast<float>(rect.width),
            static_cast<float>(rect.height),
            0.0f
        );
    }
    
    // Compute convex hull first (simplified rotating calipers)
    // For simplicity, just use the bounding rect rotated
    auto rect = boundingRect(points);
    return RotatedRect(
        Point2f(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f),
        static_cast<float>(rect.width),
        static_cast<float>(rect.height),
        0.0f
    );
}

/**
 * @brief Get minimum enclosing circle
 */
inline void minEnclosingCircle(
    const std::vector<Point>& points,
    Point2f& center,
    float& radius
) {
    if (points.empty()) {
        center = Point2f(0, 0);
        radius = 0;
        return;
    }
    
    // Simple algorithm: find farthest points
    float minX = static_cast<float>(points[0].x);
    float maxX = minX;
    float minY = static_cast<float>(points[0].y);
    float maxY = minY;
    
    for (const auto& pt : points) {
        minX = std::min(minX, static_cast<float>(pt.x));
        maxX = std::max(maxX, static_cast<float>(pt.x));
        minY = std::min(minY, static_cast<float>(pt.y));
        maxY = std::max(maxY, static_cast<float>(pt.y));
    }
    
    center.x = (minX + maxX) / 2.0f;
    center.y = (minY + maxY) / 2.0f;
    
    radius = 0.0f;
    for (const auto& pt : points) {
        float dx = pt.x - center.x;
        float dy = pt.y - center.y;
        radius = std::max(radius, std::sqrt(dx * dx + dy * dy));
    }
}

/**
 * @brief Approximate polygon with fewer vertices
 */
inline std::vector<Point> approxPolyDP(
    const std::vector<Point>& curve,
    double epsilon,
    bool closed
) {
    if (curve.size() < 3) return curve;
    
    // Douglas-Peucker algorithm
    std::vector<bool> keep(curve.size(), false);
    keep[0] = true;
    keep[curve.size() - 1] = true;
    
    std::function<void(int, int)> simplify = [&](int start, int end) {
        if (end - start < 2) return;
        
        // Find farthest point from line
        double maxDist = 0.0;
        int maxIdx = start;
        
        double x1 = curve[start].x, y1 = curve[start].y;
        double x2 = curve[end].x, y2 = curve[end].y;
        double lineLenSq = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
        
        for (int i = start + 1; i < end; ++i) {
            double dist;
            if (lineLenSq < 1e-10) {
                dist = std::sqrt((curve[i].x - x1) * (curve[i].x - x1) + 
                                (curve[i].y - y1) * (curve[i].y - y1));
            } else {
                double t = std::max(0.0, std::min(1.0,
                    ((curve[i].x - x1) * (x2 - x1) + (curve[i].y - y1) * (y2 - y1)) / lineLenSq));
                double projX = x1 + t * (x2 - x1);
                double projY = y1 + t * (y2 - y1);
                dist = std::sqrt((curve[i].x - projX) * (curve[i].x - projX) +
                                (curve[i].y - projY) * (curve[i].y - projY));
            }
            
            if (dist > maxDist) {
                maxDist = dist;
                maxIdx = i;
            }
        }
        
        if (maxDist > epsilon) {
            keep[maxIdx] = true;
            simplify(start, maxIdx);
            simplify(maxIdx, end);
        }
    };
    
    simplify(0, static_cast<int>(curve.size()) - 1);
    
    std::vector<Point> result;
    for (size_t i = 0; i < curve.size(); ++i) {
        if (keep[i]) {
            result.push_back(curve[i]);
        }
    }
    
    return result;
}

/**
 * @brief Compute convex hull
 */
inline std::vector<Point> convexHull(const std::vector<Point>& points) {
    if (points.size() < 3) return points;
    
    // Graham scan algorithm
    std::vector<Point> pts = points;
    
    // Find bottom-most point
    int minIdx = 0;
    for (size_t i = 1; i < pts.size(); ++i) {
        if (pts[i].y < pts[minIdx].y ||
            (pts[i].y == pts[minIdx].y && pts[i].x < pts[minIdx].x)) {
            minIdx = static_cast<int>(i);
        }
    }
    std::swap(pts[0], pts[minIdx]);
    Point pivot = pts[0];
    
    // Sort by polar angle
    std::sort(pts.begin() + 1, pts.end(), [&pivot](const Point& a, const Point& b) {
        int cross = (a.x - pivot.x) * (b.y - pivot.y) - (a.y - pivot.y) * (b.x - pivot.x);
        if (cross == 0) {
            int da = (a.x - pivot.x) * (a.x - pivot.x) + (a.y - pivot.y) * (a.y - pivot.y);
            int db = (b.x - pivot.x) * (b.x - pivot.x) + (b.y - pivot.y) * (b.y - pivot.y);
            return da < db;
        }
        return cross > 0;
    });
    
    // Build hull
    std::vector<Point> hull;
    for (const auto& pt : pts) {
        while (hull.size() > 1) {
            Point& p1 = hull[hull.size() - 2];
            Point& p2 = hull[hull.size() - 1];
            int cross = (p2.x - p1.x) * (pt.y - p1.y) - (p2.y - p1.y) * (pt.x - p1.x);
            if (cross <= 0) {
                hull.pop_back();
            } else {
                break;
            }
        }
        hull.push_back(pt);
    }
    
    return hull;
}

/**
 * @brief Point-in-polygon test
 */
inline double pointPolygonTest(
    const std::vector<Point>& contour,
    Point2f pt,
    bool measureDist
) {
    if (contour.size() < 3) return -1.0;
    
    // Ray casting algorithm
    int n = static_cast<int>(contour.size());
    int crossings = 0;
    
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        float y1 = static_cast<float>(contour[i].y);
        float y2 = static_cast<float>(contour[j].y);
        
        if ((y1 <= pt.y && y2 > pt.y) || (y2 <= pt.y && y1 > pt.y)) {
            float x1 = static_cast<float>(contour[i].x);
            float x2 = static_cast<float>(contour[j].x);
            float xIntersect = x1 + (pt.y - y1) / (y2 - y1) * (x2 - x1);
            if (pt.x < xIntersect) {
                ++crossings;
            }
        }
    }
    
    bool inside = (crossings % 2) == 1;
    
    if (!measureDist) {
        return inside ? 1.0 : -1.0;
    }
    
    // Find minimum distance to contour
    double minDist = 1e10;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        double x1 = contour[i].x, y1 = contour[i].y;
        double x2 = contour[j].x, y2 = contour[j].y;
        
        double dx = x2 - x1, dy = y2 - y1;
        double lenSq = dx * dx + dy * dy;
        
        double t = std::max(0.0, std::min(1.0,
            ((pt.x - x1) * dx + (pt.y - y1) * dy) / lenSq));
        
        double projX = x1 + t * dx;
        double projY = y1 + t * dy;
        
        double dist = std::sqrt((pt.x - projX) * (pt.x - projX) + 
                               (pt.y - projY) * (pt.y - projY));
        minDist = std::min(minDist, dist);
    }
    
    return inside ? minDist : -minDist;
}

/**
 * @brief Calculate image moments
 */
inline Moments moments(
    const std::vector<Point>& contour,
    bool binaryImage = false
) {
    Moments m;
    
    if (contour.size() < 3) return m;
    
    // Calculate raw moments
    for (const auto& pt : contour) {
        double x = pt.x, y = pt.y;
        m.m00 += 1;
        m.m10 += x;
        m.m01 += y;
        m.m20 += x * x;
        m.m11 += x * y;
        m.m02 += y * y;
        m.m30 += x * x * x;
        m.m21 += x * x * y;
        m.m12 += x * y * y;
        m.m03 += y * y * y;
    }
    
    // Calculate central moments
    if (m.m00 > 0) {
        double cx = m.m10 / m.m00;
        double cy = m.m01 / m.m00;
        
        m.mu20 = m.m20 - cx * m.m10;
        m.mu11 = m.m11 - cx * m.m01;
        m.mu02 = m.m02 - cy * m.m01;
        m.mu30 = m.m30 - 3 * cx * m.m20 + 2 * cx * cx * m.m10;
        m.mu21 = m.m21 - 2 * cx * m.m11 - cy * m.m20 + 2 * cx * cx * m.m01;
        m.mu12 = m.m12 - 2 * cy * m.m11 - cx * m.m02 + 2 * cy * cy * m.m10;
        m.mu03 = m.m03 - 3 * cy * m.m02 + 2 * cy * cy * m.m01;
        
        // Normalized central moments
        double n = std::pow(m.m00, 2.0 / 2 + 1);
        m.nu20 = m.mu20 / n;
        m.nu11 = m.mu11 / n;
        m.nu02 = m.mu02 / n;
        
        n = std::pow(m.m00, 3.0 / 2 + 1);
        m.nu30 = m.mu30 / n;
        m.nu21 = m.mu21 / n;
        m.nu12 = m.mu12 / n;
        m.nu03 = m.mu03 / n;
    }
    
    return m;
}

/**
 * @brief Check if contour is convex
 */
inline bool isContourConvex(const std::vector<Point>& contour) {
    if (contour.size() < 3) return true;
    
    int n = static_cast<int>(contour.size());
    int sign = 0;
    
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        int k = (i + 2) % n;
        
        int cross = (contour[j].x - contour[i].x) * (contour[k].y - contour[j].y) -
                    (contour[j].y - contour[i].y) * (contour[k].x - contour[j].x);
        
        if (cross != 0) {
            int newSign = (cross > 0) ? 1 : -1;
            if (sign != 0 && newSign != sign) {
                return false;
            }
            sign = newSign;
        }
    }
    
    return true;
}

/**
 * @brief Match contour shapes using Hu moments
 */
inline double matchShapes(
    const std::vector<Point>& contour1,
    const std::vector<Point>& contour2,
    int method = 1
) {
    Moments m1 = moments(contour1);
    Moments m2 = moments(contour2);
    
    // Calculate Hu moments
    auto huMoments = [](const Moments& m) -> std::array<double, 7> {
        return {
            m.nu20 + m.nu02,
            (m.nu20 - m.nu02) * (m.nu20 - m.nu02) + 4 * m.nu11 * m.nu11,
            (m.nu30 - 3 * m.nu12) * (m.nu30 - 3 * m.nu12) + (3 * m.nu21 - m.nu03) * (3 * m.nu21 - m.nu03),
            (m.nu30 + m.nu12) * (m.nu30 + m.nu12) + (m.nu21 + m.nu03) * (m.nu21 + m.nu03),
            (m.nu30 - 3 * m.nu12) * (m.nu30 + m.nu12) * ((m.nu30 + m.nu12) * (m.nu30 + m.nu12) - 3 * (m.nu21 + m.nu03) * (m.nu21 + m.nu03)) +
            (3 * m.nu21 - m.nu03) * (m.nu21 + m.nu03) * (3 * (m.nu30 + m.nu12) * (m.nu30 + m.nu12) - (m.nu21 + m.nu03) * (m.nu21 + m.nu03)),
            (m.nu20 - m.nu02) * ((m.nu30 + m.nu12) * (m.nu30 + m.nu12) - (m.nu21 + m.nu03) * (m.nu21 + m.nu03)) +
            4 * m.nu11 * (m.nu30 + m.nu12) * (m.nu21 + m.nu03),
            (3 * m.nu21 - m.nu03) * (m.nu30 + m.nu12) * ((m.nu30 + m.nu12) * (m.nu30 + m.nu12) - 3 * (m.nu21 + m.nu03) * (m.nu21 + m.nu03)) -
            (m.nu30 - 3 * m.nu12) * (m.nu21 + m.nu03) * (3 * (m.nu30 + m.nu12) * (m.nu30 + m.nu12) - (m.nu21 + m.nu03) * (m.nu21 + m.nu03))
        };
    };
    
    auto hu1 = huMoments(m1);
    auto hu2 = huMoments(m2);
    
    double result = 0.0;
    for (int i = 0; i < 7; ++i) {
        double a = (std::abs(hu1[i]) > 0) ? std::copysign(std::log10(std::abs(hu1[i])), hu1[i]) : 0;
        double b = (std::abs(hu2[i]) > 0) ? std::copysign(std::log10(std::abs(hu2[i])), hu2[i]) : 0;
        
        if (method == 1) {
            result += std::abs(1.0 / a - 1.0 / b);
        } else if (method == 2) {
            result += std::abs(a - b);
        } else {
            result = std::max(result, std::abs(a - b));
        }
    }
    
    return result;
}

} // namespace imgproc
} // namespace neurova

#endif // NEUROVA_IMGPROC_CONTOURS_HPP
