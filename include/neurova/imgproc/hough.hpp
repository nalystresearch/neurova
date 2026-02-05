// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file hough.hpp
 * @brief Hough transform for line and circle detection
 */

#ifndef NEUROVA_IMGPROC_HOUGH_HPP
#define NEUROVA_IMGPROC_HOUGH_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <array>

namespace neurova {
namespace imgproc {

// Hough modes
constexpr int HOUGH_STANDARD = 0;
constexpr int HOUGH_PROBABILISTIC = 1;
constexpr int HOUGH_MULTI_SCALE = 2;
constexpr int HOUGH_GRADIENT = 3;
constexpr int HOUGH_GRADIENT_ALT = 4;

/**
 * @brief Line representation (rho, theta) for standard Hough
 */
struct HoughLine {
    float rho;
    float theta;
    int votes;
    
    HoughLine() : rho(0), theta(0), votes(0) {}
    HoughLine(float r, float t, int v = 0) : rho(r), theta(t), votes(v) {}
};

/**
 * @brief Line segment for probabilistic Hough
 */
struct LineSegment {
    int x1, y1, x2, y2;
    
    LineSegment() : x1(0), y1(0), x2(0), y2(0) {}
    LineSegment(int x1_, int y1_, int x2_, int y2_) 
        : x1(x1_), y1(y1_), x2(x2_), y2(y2_) {}
    
    double length() const {
        double dx = x2 - x1;
        double dy = y2 - y1;
        return std::sqrt(dx * dx + dy * dy);
    }
};

/**
 * @brief Circle representation
 */
struct HoughCircle {
    float x, y, radius;
    int votes;
    
    HoughCircle() : x(0), y(0), radius(0), votes(0) {}
    HoughCircle(float x_, float y_, float r_, int v = 0) 
        : x(x_), y(y_), radius(r_), votes(v) {}
};

/**
 * @brief Standard Hough line transform
 */
inline std::vector<HoughLine> HoughLines(
    const float* edges, int width, int height,
    double rho = 1.0,
    double theta = M_PI / 180.0,
    int threshold = 100
) {
    // Calculate accumulator dimensions
    double diagonal = std::sqrt(static_cast<double>(width * width + height * height));
    int numRho = static_cast<int>(2.0 * diagonal / rho) + 1;
    int numTheta = static_cast<int>(M_PI / theta);
    
    // Create accumulator
    std::vector<int> accumulator(numRho * numTheta, 0);
    
    // Precompute sin/cos tables
    std::vector<double> cosTable(numTheta);
    std::vector<double> sinTable(numTheta);
    for (int t = 0; t < numTheta; ++t) {
        double angle = t * theta;
        cosTable[t] = std::cos(angle);
        sinTable[t] = std::sin(angle);
    }
    
    // Vote
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (edges[y * width + x] > 127) {
                for (int t = 0; t < numTheta; ++t) {
                    double r = x * cosTable[t] + y * sinTable[t];
                    int rIdx = static_cast<int>((r + diagonal) / rho);
                    if (rIdx >= 0 && rIdx < numRho) {
                        accumulator[rIdx * numTheta + t]++;
                    }
                }
            }
        }
    }
    
    // Extract lines above threshold
    std::vector<HoughLine> lines;
    
    for (int r = 0; r < numRho; ++r) {
        for (int t = 0; t < numTheta; ++t) {
            int votes = accumulator[r * numTheta + t];
            if (votes >= threshold) {
                // Check if local maximum (3x3 neighborhood)
                bool isMax = true;
                for (int dr = -1; dr <= 1 && isMax; ++dr) {
                    for (int dt = -1; dt <= 1 && isMax; ++dt) {
                        if (dr == 0 && dt == 0) continue;
                        int nr = r + dr;
                        int nt = t + dt;
                        if (nr >= 0 && nr < numRho && nt >= 0 && nt < numTheta) {
                            if (accumulator[nr * numTheta + nt] > votes) {
                                isMax = false;
                            }
                        }
                    }
                }
                
                if (isMax) {
                    float lineRho = static_cast<float>(r * rho - diagonal);
                    float lineTheta = static_cast<float>(t * theta);
                    lines.push_back(HoughLine(lineRho, lineTheta, votes));
                }
            }
        }
    }
    
    // Sort by votes
    std::sort(lines.begin(), lines.end(), [](const HoughLine& a, const HoughLine& b) {
        return a.votes > b.votes;
    });
    
    return lines;
}

/**
 * @brief Probabilistic Hough line transform
 */
inline std::vector<LineSegment> HoughLinesP(
    const float* edges, int width, int height,
    double rho = 1.0,
    double theta = M_PI / 180.0,
    int threshold = 50,
    double minLineLength = 30.0,
    double maxLineGap = 10.0
) {
    // Collect edge points
    std::vector<std::pair<int, int>> edgePoints;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (edges[y * width + x] > 127) {
                edgePoints.push_back({x, y});
            }
        }
    }
    
    // Accumulator dimensions
    double diagonal = std::sqrt(static_cast<double>(width * width + height * height));
    int numRho = static_cast<int>(2.0 * diagonal / rho) + 1;
    int numTheta = static_cast<int>(M_PI / theta);
    
    // Precompute sin/cos
    std::vector<double> cosTable(numTheta);
    std::vector<double> sinTable(numTheta);
    for (int t = 0; t < numTheta; ++t) {
        double angle = t * theta;
        cosTable[t] = std::cos(angle);
        sinTable[t] = std::sin(angle);
    }
    
    std::vector<LineSegment> lines;
    std::vector<bool> used(edgePoints.size(), false);
    std::vector<int> accumulator(numRho * numTheta, 0);
    
    // Process points randomly
    std::vector<size_t> indices(edgePoints.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    
    for (size_t idx : indices) {
        if (used[idx]) continue;
        
        int px = edgePoints[idx].first;
        int py = edgePoints[idx].second;
        
        // Vote
        std::fill(accumulator.begin(), accumulator.end(), 0);
        for (int t = 0; t < numTheta; ++t) {
            double r = px * cosTable[t] + py * sinTable[t];
            int rIdx = static_cast<int>((r + diagonal) / rho);
            if (rIdx >= 0 && rIdx < numRho) {
                accumulator[rIdx * numTheta + t]++;
            }
        }
        
        // Find maximum
        int maxVotes = 0, maxR = 0, maxT = 0;
        for (int r = 0; r < numRho; ++r) {
            for (int t = 0; t < numTheta; ++t) {
                if (accumulator[r * numTheta + t] > maxVotes) {
                    maxVotes = accumulator[r * numTheta + t];
                    maxR = r;
                    maxT = t;
                }
            }
        }
        
        if (maxVotes >= threshold) {
            // Find all points on this line
            double lineRho = maxR * rho - diagonal;
            double ct = cosTable[maxT];
            double st = sinTable[maxT];
            
            std::vector<std::pair<int, int>> linePoints;
            for (size_t i = 0; i < edgePoints.size(); ++i) {
                if (used[i]) continue;
                int x = edgePoints[i].first;
                int y = edgePoints[i].second;
                double r = x * ct + y * st;
                if (std::abs(r - lineRho) < rho * 2) {
                    linePoints.push_back({x, y});
                }
            }
            
            if (linePoints.size() < 2) continue;
            
            // Sort points along line direction
            if (std::abs(ct) > std::abs(st)) {
                std::sort(linePoints.begin(), linePoints.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            } else {
                std::sort(linePoints.begin(), linePoints.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; });
            }
            
            // Extract line segments
            int segStart = 0;
            for (size_t i = 1; i < linePoints.size(); ++i) {
                double dx = linePoints[i].first - linePoints[i - 1].first;
                double dy = linePoints[i].second - linePoints[i - 1].second;
                double gap = std::sqrt(dx * dx + dy * dy);
                
                if (gap > maxLineGap) {
                    // End segment
                    double segDx = linePoints[i - 1].first - linePoints[segStart].first;
                    double segDy = linePoints[i - 1].second - linePoints[segStart].second;
                    double segLen = std::sqrt(segDx * segDx + segDy * segDy);
                    
                    if (segLen >= minLineLength) {
                        lines.push_back(LineSegment(
                            linePoints[segStart].first, linePoints[segStart].second,
                            linePoints[i - 1].first, linePoints[i - 1].second
                        ));
                    }
                    segStart = static_cast<int>(i);
                }
            }
            
            // Last segment
            double segDx = linePoints.back().first - linePoints[segStart].first;
            double segDy = linePoints.back().second - linePoints[segStart].second;
            double segLen = std::sqrt(segDx * segDx + segDy * segDy);
            
            if (segLen >= minLineLength) {
                lines.push_back(LineSegment(
                    linePoints[segStart].first, linePoints[segStart].second,
                    linePoints.back().first, linePoints.back().second
                ));
            }
            
            // Mark points as used
            for (const auto& pt : linePoints) {
                for (size_t i = 0; i < edgePoints.size(); ++i) {
                    if (edgePoints[i] == pt) {
                        used[i] = true;
                        break;
                    }
                }
            }
        }
        
        used[idx] = true;
    }
    
    return lines;
}

/**
 * @brief Hough circle transform
 */
inline std::vector<HoughCircle> HoughCircles(
    const float* edges, int width, int height,
    int method = HOUGH_GRADIENT,
    double dp = 1.0,
    double minDist = 20.0,
    double param1 = 100.0,
    double param2 = 30.0,
    int minRadius = 0,
    int maxRadius = 0
) {
    std::vector<HoughCircle> circles;
    
    if (maxRadius <= 0) {
        maxRadius = std::min(width, height) / 2;
    }
    
    // Downscaled accumulator dimensions
    int accWidth = static_cast<int>(width / dp);
    int accHeight = static_cast<int>(height / dp);
    
    // Create 2D accumulator for center voting
    std::vector<int> centerAcc(accWidth * accHeight, 0);
    
    // Compute gradient direction at edge points
    std::vector<std::tuple<int, int, double>> edgeGrad;  // x, y, angle
    
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            if (edges[y * width + x] > param1 / 2) {
                // Sobel gradient
                double gx = -edges[(y - 1) * width + (x - 1)] - 2 * edges[y * width + (x - 1)] - edges[(y + 1) * width + (x - 1)]
                          + edges[(y - 1) * width + (x + 1)] + 2 * edges[y * width + (x + 1)] + edges[(y + 1) * width + (x + 1)];
                double gy = -edges[(y - 1) * width + (x - 1)] - 2 * edges[(y - 1) * width + x] - edges[(y - 1) * width + (x + 1)]
                          + edges[(y + 1) * width + (x - 1)] + 2 * edges[(y + 1) * width + x] + edges[(y + 1) * width + (x + 1)];
                
                double mag = std::sqrt(gx * gx + gy * gy);
                if (mag > 1e-6) {
                    double angle = std::atan2(gy, gx);
                    edgeGrad.push_back({x, y, angle});
                }
            }
        }
    }
    
    // Vote for centers along gradient direction
    for (const auto& [x, y, angle] : edgeGrad) {
        double cos_a = std::cos(angle);
        double sin_a = std::sin(angle);
        
        for (int r = minRadius; r <= maxRadius; ++r) {
            // Vote in both directions
            for (int sign = -1; sign <= 1; sign += 2) {
                int cx = static_cast<int>((x + sign * r * cos_a) / dp);
                int cy = static_cast<int>((y + sign * r * sin_a) / dp);
                
                if (cx >= 0 && cx < accWidth && cy >= 0 && cy < accHeight) {
                    centerAcc[cy * accWidth + cx]++;
                }
            }
        }
    }
    
    // Find center candidates
    std::vector<std::pair<int, int>> centerCandidates;
    
    for (int y = 0; y < accHeight; ++y) {
        for (int x = 0; x < accWidth; ++x) {
            int votes = centerAcc[y * accWidth + x];
            if (votes >= param2) {
                // Local maximum check
                bool isMax = true;
                int windowSize = static_cast<int>(minDist / dp / 2);
                
                for (int dy = -windowSize; dy <= windowSize && isMax; ++dy) {
                    for (int dx = -windowSize; dx <= windowSize && isMax; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < accWidth && ny >= 0 && ny < accHeight) {
                            if (centerAcc[ny * accWidth + nx] > votes) {
                                isMax = false;
                            }
                        }
                    }
                }
                
                if (isMax) {
                    centerCandidates.push_back({
                        static_cast<int>(x * dp),
                        static_cast<int>(y * dp)
                    });
                }
            }
        }
    }
    
    // For each center, find best radius
    for (const auto& [cx, cy] : centerCandidates) {
        std::vector<int> radiusHist(maxRadius - minRadius + 1, 0);
        
        for (const auto& [ex, ey, angle] : edgeGrad) {
            double dist = std::sqrt((ex - cx) * (ex - cx) + (ey - cy) * (ey - cy));
            int rIdx = static_cast<int>(dist) - minRadius;
            if (rIdx >= 0 && rIdx < static_cast<int>(radiusHist.size())) {
                radiusHist[rIdx]++;
            }
        }
        
        // Find best radius
        int maxRadiusVotes = 0;
        int bestRadius = minRadius;
        for (size_t i = 0; i < radiusHist.size(); ++i) {
            if (radiusHist[i] > maxRadiusVotes) {
                maxRadiusVotes = radiusHist[i];
                bestRadius = static_cast<int>(i) + minRadius;
            }
        }
        
        if (maxRadiusVotes >= param2 / 2) {
            // Check if not too close to existing circle
            bool tooClose = false;
            for (const auto& c : circles) {
                double dist = std::sqrt((cx - c.x) * (cx - c.x) + (cy - c.y) * (cy - c.y));
                if (dist < minDist) {
                    tooClose = true;
                    break;
                }
            }
            
            if (!tooClose) {
                circles.push_back(HoughCircle(
                    static_cast<float>(cx),
                    static_cast<float>(cy),
                    static_cast<float>(bestRadius),
                    maxRadiusVotes
                ));
            }
        }
    }
    
    // Sort by votes
    std::sort(circles.begin(), circles.end(), [](const HoughCircle& a, const HoughCircle& b) {
        return a.votes > b.votes;
    });
    
    return circles;
}

/**
 * @brief Convert line (rho, theta) to two points
 */
inline void lineRhoThetaToPoints(
    float rho, float theta,
    int width, int height,
    int& x1, int& y1, int& x2, int& y2
) {
    double ct = std::cos(theta);
    double st = std::sin(theta);
    double x0 = ct * rho;
    double y0 = st * rho;
    
    // Extend line to image boundaries
    double scale = std::max(width, height);
    x1 = static_cast<int>(x0 - scale * st);
    y1 = static_cast<int>(y0 + scale * ct);
    x2 = static_cast<int>(x0 + scale * st);
    y2 = static_cast<int>(y0 - scale * ct);
}

/**
 * @brief Get intersection of two lines
 */
inline bool getLineIntersection(
    const HoughLine& line1,
    const HoughLine& line2,
    float& x, float& y
) {
    double ct1 = std::cos(line1.theta);
    double st1 = std::sin(line1.theta);
    double ct2 = std::cos(line2.theta);
    double st2 = std::sin(line2.theta);
    
    double det = ct1 * st2 - ct2 * st1;
    if (std::abs(det) < 1e-10) {
        return false;  // Parallel lines
    }
    
    x = static_cast<float>((st2 * line1.rho - st1 * line2.rho) / det);
    y = static_cast<float>((ct1 * line2.rho - ct2 * line1.rho) / det);
    
    return true;
}

/**
 * @brief Detect line segments using LSD algorithm (simplified)
 */
inline std::vector<LineSegment> createLineSegmentDetector(
    const float* gray, int width, int height,
    double scale = 0.8,
    double sigmaScale = 0.6,
    double quant = 2.0,
    double angTh = 22.5,
    double logEps = 0.0,
    double densityTh = 0.7
) {
    std::vector<LineSegment> segments;
    
    // Compute gradient
    std::vector<double> gradX(width * height);
    std::vector<double> gradY(width * height);
    std::vector<double> gradMag(width * height);
    std::vector<double> gradAng(width * height);
    
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            double gx = gray[y * width + (x + 1)] - gray[y * width + (x - 1)];
            double gy = gray[(y + 1) * width + x] - gray[(y - 1) * width + x];
            
            int idx = y * width + x;
            gradX[idx] = gx;
            gradY[idx] = gy;
            gradMag[idx] = std::sqrt(gx * gx + gy * gy);
            gradAng[idx] = std::atan2(gy, gx);
        }
    }
    
    // Find gradient magnitude threshold
    double maxMag = 0;
    for (int i = 0; i < width * height; ++i) {
        maxMag = std::max(maxMag, gradMag[i]);
    }
    double threshold = maxMag * quant / 255.0;
    
    // Region growing
    std::vector<bool> used(width * height, false);
    double angThRad = angTh * M_PI / 180.0;
    
    // Sort pixels by gradient magnitude (descending)
    std::vector<std::pair<double, int>> sortedPixels;
    for (int i = 0; i < width * height; ++i) {
        if (gradMag[i] > threshold) {
            sortedPixels.push_back({gradMag[i], i});
        }
    }
    std::sort(sortedPixels.begin(), sortedPixels.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    for (const auto& [mag, seedIdx] : sortedPixels) {
        if (used[seedIdx]) continue;
        
        int seedX = seedIdx % width;
        int seedY = seedIdx / width;
        double seedAng = gradAng[seedIdx];
        
        // Grow region
        std::vector<std::pair<int, int>> region;
        std::vector<int> queue;
        queue.push_back(seedIdx);
        used[seedIdx] = true;
        
        while (!queue.empty()) {
            int idx = queue.back();
            queue.pop_back();
            int x = idx % width;
            int y = idx / width;
            region.push_back({x, y});
            
            // Check 8-neighbors
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                    
                    int nidx = ny * width + nx;
                    if (used[nidx]) continue;
                    if (gradMag[nidx] <= threshold) continue;
                    
                    // Check angle
                    double angDiff = std::abs(gradAng[nidx] - seedAng);
                    if (angDiff > M_PI) angDiff = 2 * M_PI - angDiff;
                    if (angDiff < angThRad || angDiff > M_PI - angThRad) {
                        used[nidx] = true;
                        queue.push_back(nidx);
                    }
                }
            }
        }
        
        if (region.size() < 5) continue;
        
        // Fit line to region using SVD (simplified: use endpoints)
        int minX = width, maxX = 0, minY = height, maxY = 0;
        for (const auto& [x, y] : region) {
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }
        
        double lineLen = std::sqrt((maxX - minX) * (maxX - minX) + (maxY - minY) * (maxY - minY));
        if (lineLen > 5) {
            segments.push_back(LineSegment(minX, minY, maxX, maxY));
        }
    }
    
    return segments;
}

} // namespace imgproc
} // namespace neurova

#endif // NEUROVA_IMGPROC_HOUGH_HPP
