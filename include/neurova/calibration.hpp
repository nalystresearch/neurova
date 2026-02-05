// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file calibration.hpp
 * @brief Camera calibration and pose estimation
 * 
 * Neurova implementation of camera calibration, distortion correction,
 * and pose estimation algorithms.
 */

#pragma once

#include "core/image.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace neurova {
namespace calibration {

// ============================================================================
// Types and Structures
// ============================================================================

/**
 * @brief Camera intrinsic matrix (3x3)
 */
struct CameraMatrix {
    float fx = 1.0f;  // Focal length x
    float fy = 1.0f;  // Focal length y
    float cx = 0.0f;  // Principal point x
    float cy = 0.0f;  // Principal point y
    
    std::array<float, 9> to_array() const {
        return {fx, 0, cx,
                0, fy, cy,
                0, 0, 1};
    }
    
    static CameraMatrix from_array(const std::array<float, 9>& arr) {
        CameraMatrix m;
        m.fx = arr[0];
        m.fy = arr[4];
        m.cx = arr[2];
        m.cy = arr[5];
        return m;
    }
};

/**
 * @brief Distortion coefficients (radial and tangential)
 */
struct DistortionCoeffs {
    float k1 = 0;  // Radial 1
    float k2 = 0;  // Radial 2
    float p1 = 0;  // Tangential 1
    float p2 = 0;  // Tangential 2
    float k3 = 0;  // Radial 3
    float k4 = 0;  // Radial 4 (fisheye)
    float k5 = 0;  // Radial 5 (fisheye)
    float k6 = 0;  // Radial 6 (fisheye)
    
    std::vector<float> to_vector() const {
        return {k1, k2, p1, p2, k3, k4, k5, k6};
    }
};

/**
 * @brief 3D point
 */
struct Point3f {
    float x = 0, y = 0, z = 0;
    
    Point3f() = default;
    Point3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

/**
 * @brief 2D point
 */
struct Point2f {
    float x = 0, y = 0;
    
    Point2f() = default;
    Point2f(float x_, float y_) : x(x_), y(y_) {}
};

/**
 * @brief Rotation-Translation transformation
 */
struct RigidTransform {
    std::array<float, 9> rotation = {1,0,0, 0,1,0, 0,0,1};  // 3x3 rotation matrix
    std::array<float, 3> translation = {0, 0, 0};           // Translation vector
    
    // Convert to 4x4 homogeneous matrix
    std::array<float, 16> to_matrix4x4() const {
        return {
            rotation[0], rotation[1], rotation[2], translation[0],
            rotation[3], rotation[4], rotation[5], translation[1],
            rotation[6], rotation[7], rotation[8], translation[2],
            0, 0, 0, 1
        };
    }
};

/**
 * @brief Calibration result
 */
struct CalibrationResult {
    CameraMatrix camera_matrix;
    DistortionCoeffs dist_coeffs;
    std::vector<RigidTransform> extrinsics;  // Per-image transforms
    float rms_error = 0;
    bool success = false;
};

// ============================================================================
// Calibration Pattern Detection
// ============================================================================

/**
 * @brief Find chessboard corners
 */
inline bool find_chessboard_corners(const Image& image, int pattern_width, int pattern_height,
                                     std::vector<Point2f>& corners) {
    corners.clear();
    
    // Convert to grayscale if needed
    Image gray(image.width(), image.height(), 1);
    if (image.channels() >= 3) {
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x) {
                gray.at(x, y, 0) = 0.299f * image.at(x, y, 0) + 
                                  0.587f * image.at(x, y, 1) + 
                                  0.114f * image.at(x, y, 2);
            }
        }
    } else {
        gray = image;
    }
    
    // Compute Harris corners
    Image harris(image.width(), image.height(), 1);
    float k = 0.04f;
    int window = 3;
    
    for (int y = window; y < image.height() - window; ++y) {
        for (int x = window; x < image.width() - window; ++x) {
            float Ix = (gray.at(x+1, y, 0) - gray.at(x-1, y, 0)) / 2.0f;
            float Iy = (gray.at(x, y+1, 0) - gray.at(x, y-1, 0)) / 2.0f;
            
            float Ixx = 0, Iyy = 0, Ixy = 0;
            for (int dy = -window; dy <= window; ++dy) {
                for (int dx = -window; dx <= window; ++dx) {
                    float ix = (gray.at(x+dx+1, y+dy, 0) - gray.at(x+dx-1, y+dy, 0)) / 2.0f;
                    float iy = (gray.at(x+dx, y+dy+1, 0) - gray.at(x+dx, y+dy-1, 0)) / 2.0f;
                    Ixx += ix * ix;
                    Iyy += iy * iy;
                    Ixy += ix * iy;
                }
            }
            
            float det = Ixx * Iyy - Ixy * Ixy;
            float trace = Ixx + Iyy;
            harris.at(x, y, 0) = det - k * trace * trace;
        }
    }
    
    // Non-maximum suppression and threshold
    std::vector<Point2f> candidates;
    float threshold = 0.01f * 255 * 255;
    
    for (int y = 5; y < image.height() - 5; ++y) {
        for (int x = 5; x < image.width() - 5; ++x) {
            float val = harris.at(x, y, 0);
            if (val > threshold) {
                bool is_max = true;
                for (int dy = -2; dy <= 2 && is_max; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        if (harris.at(x+dx, y+dy, 0) > val) {
                            is_max = false;
                            break;
                        }
                    }
                }
                if (is_max) {
                    candidates.push_back(Point2f(static_cast<float>(x), static_cast<float>(y)));
                }
            }
        }
    }
    
    // Try to organize corners into a grid
    // This is a simplified version - real implementation would be more robust
    if (candidates.size() >= static_cast<size_t>(pattern_width * pattern_height)) {
        // Sort by y then x
        std::sort(candidates.begin(), candidates.end(), 
                  [](const Point2f& a, const Point2f& b) {
                      if (std::abs(a.y - b.y) > 20) return a.y < b.y;
                      return a.x < b.x;
                  });
        
        // Take first pattern_width * pattern_height corners
        for (int i = 0; i < pattern_width * pattern_height && i < static_cast<int>(candidates.size()); ++i) {
            corners.push_back(candidates[i]);
        }
        
        return corners.size() == static_cast<size_t>(pattern_width * pattern_height);
    }
    
    return false;
}

/**
 * @brief Refine corner positions to sub-pixel accuracy
 */
inline void corner_sub_pix(const Image& image, std::vector<Point2f>& corners,
                           int window_size = 5, int max_iter = 30, float epsilon = 0.01f) {
    // Convert to grayscale if needed
    Image gray(image.width(), image.height(), 1);
    if (image.channels() >= 3) {
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x) {
                gray.at(x, y, 0) = 0.299f * image.at(x, y, 0) + 
                                  0.587f * image.at(x, y, 1) + 
                                  0.114f * image.at(x, y, 2);
            }
        }
    } else {
        gray = image;
    }
    
    for (auto& corner : corners) {
        for (int iter = 0; iter < max_iter; ++iter) {
            float sum_gxx = 0, sum_gyy = 0, sum_gxy = 0;
            float sum_gx_dx = 0, sum_gy_dy = 0;
            
            int cx = static_cast<int>(corner.x);
            int cy = static_cast<int>(corner.y);
            
            for (int dy = -window_size; dy <= window_size; ++dy) {
                for (int dx = -window_size; dx <= window_size; ++dx) {
                    int x = cx + dx;
                    int y = cy + dy;
                    
                    if (x < 1 || x >= image.width() - 1 || y < 1 || y >= image.height() - 1)
                        continue;
                    
                    float gx = (gray.at(x+1, y, 0) - gray.at(x-1, y, 0)) / 2.0f;
                    float gy = (gray.at(x, y+1, 0) - gray.at(x, y-1, 0)) / 2.0f;
                    
                    sum_gxx += gx * gx;
                    sum_gyy += gy * gy;
                    sum_gxy += gx * gy;
                    sum_gx_dx += gx * dx;
                    sum_gy_dy += gy * dy;
                }
            }
            
            // Solve 2x2 system
            float det = sum_gxx * sum_gyy - sum_gxy * sum_gxy;
            if (std::abs(det) < 1e-6f) break;
            
            float dx = -(sum_gyy * sum_gx_dx - sum_gxy * sum_gy_dy) / det;
            float dy = -(sum_gxx * sum_gy_dy - sum_gxy * sum_gx_dx) / det;
            
            corner.x += dx;
            corner.y += dy;
            
            if (dx * dx + dy * dy < epsilon * epsilon) break;
        }
    }
}

/**
 * @brief Draw detected chessboard corners on image
 */
inline void draw_chessboard_corners(Image& image, int pattern_width, int pattern_height,
                                     const std::vector<Point2f>& corners, bool found) {
    if (corners.empty()) return;
    
    // Draw corners
    for (size_t i = 0; i < corners.size(); ++i) {
        int x = static_cast<int>(corners[i].x);
        int y = static_cast<int>(corners[i].y);
        
        // Color gradient from red to blue
        float t = static_cast<float>(i) / corners.size();
        float r = (1.0f - t) * 255;
        float g = 0;
        float b = t * 255;
        
        // Draw circle
        int radius = found ? 5 : 3;
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
                        } else {
                            image.at(px, py, 0) = 255;
                        }
                    }
                }
            }
        }
    }
    
    // Draw lines between adjacent corners
    if (found) {
        for (int row = 0; row < pattern_height; ++row) {
            for (int col = 0; col < pattern_width - 1; ++col) {
                int i1 = row * pattern_width + col;
                int i2 = row * pattern_width + col + 1;
                
                // Draw line (Bresenham)
                int x0 = static_cast<int>(corners[i1].x);
                int y0 = static_cast<int>(corners[i1].y);
                int x1 = static_cast<int>(corners[i2].x);
                int y1 = static_cast<int>(corners[i2].y);
                
                int dx = std::abs(x1 - x0);
                int dy = std::abs(y1 - y0);
                int sx = x0 < x1 ? 1 : -1;
                int sy = y0 < y1 ? 1 : -1;
                int err = dx - dy;
                
                while (true) {
                    if (x0 >= 0 && x0 < image.width() && y0 >= 0 && y0 < image.height()) {
                        if (image.channels() >= 3) {
                            image.at(x0, y0, 1) = 255;  // Green line
                        }
                    }
                    
                    if (x0 == x1 && y0 == y1) break;
                    int e2 = 2 * err;
                    if (e2 > -dy) { err -= dy; x0 += sx; }
                    if (e2 < dx) { err += dx; y0 += sy; }
                }
            }
        }
    }
}

// ============================================================================
// Camera Calibration
// ============================================================================

/**
 * @brief Generate object points for a calibration pattern
 */
inline std::vector<Point3f> create_pattern_points(int pattern_width, int pattern_height,
                                                   float square_size = 1.0f) {
    std::vector<Point3f> points;
    for (int row = 0; row < pattern_height; ++row) {
        for (int col = 0; col < pattern_width; ++col) {
            points.push_back(Point3f(col * square_size, row * square_size, 0));
        }
    }
    return points;
}

/**
 * @brief Estimate initial camera matrix from image size
 */
inline CameraMatrix init_camera_matrix(int image_width, int image_height) {
    CameraMatrix matrix;
    matrix.fx = static_cast<float>(std::max(image_width, image_height));
    matrix.fy = matrix.fx;
    matrix.cx = image_width / 2.0f;
    matrix.cy = image_height / 2.0f;
    return matrix;
}

/**
 * @brief Calibrate camera from multiple images
 * 
 * Simplified Zhang's method implementation
 */
inline CalibrationResult calibrate_camera(
    const std::vector<std::vector<Point3f>>& object_points,
    const std::vector<std::vector<Point2f>>& image_points,
    int image_width, int image_height) {
    
    CalibrationResult result;
    
    if (object_points.empty() || object_points.size() != image_points.size()) {
        return result;
    }
    
    // Initialize camera matrix
    result.camera_matrix = init_camera_matrix(image_width, image_height);
    
    // Simplified calibration using homographies
    // (Full implementation would use Levenberg-Marquardt optimization)
    
    std::vector<std::array<float, 9>> homographies;
    
    for (size_t img = 0; img < object_points.size(); ++img) {
        const auto& obj_pts = object_points[img];
        const auto& img_pts = image_points[img];
        
        if (obj_pts.size() < 4) continue;
        
        // Compute homography using DLT (simplified)
        // H: 3x3 matrix mapping object points to image points
        std::array<float, 9> H = {1,0,0, 0,1,0, 0,0,1};
        
        // Centroid for normalization
        float obj_cx = 0, obj_cy = 0, img_cx = 0, img_cy = 0;
        for (size_t i = 0; i < obj_pts.size(); ++i) {
            obj_cx += obj_pts[i].x;
            obj_cy += obj_pts[i].y;
            img_cx += img_pts[i].x;
            img_cy += img_pts[i].y;
        }
        obj_cx /= obj_pts.size();
        obj_cy /= obj_pts.size();
        img_cx /= img_pts.size();
        img_cy /= img_pts.size();
        
        // Simple similarity transform estimation
        float scale = 0;
        float rotation = 0;
        
        for (size_t i = 0; i < obj_pts.size(); ++i) {
            float ox = obj_pts[i].x - obj_cx;
            float oy = obj_pts[i].y - obj_cy;
            float ix = img_pts[i].x - img_cx;
            float iy = img_pts[i].y - img_cy;
            
            float od = std::sqrt(ox*ox + oy*oy);
            float id = std::sqrt(ix*ix + iy*iy);
            
            if (od > 1e-6f) {
                scale += id / od;
            }
        }
        scale /= obj_pts.size();
        
        H[0] = scale; H[1] = 0; H[2] = img_cx - scale * obj_cx;
        H[3] = 0; H[4] = scale; H[5] = img_cy - scale * obj_cy;
        H[6] = 0; H[7] = 0; H[8] = 1;
        
        homographies.push_back(H);
        
        // Extract rotation and translation (simplified)
        RigidTransform transform;
        float f = result.camera_matrix.fx;
        
        transform.rotation = {
            H[0] / scale, H[1] / scale, H[2] / f,
            H[3] / scale, H[4] / scale, H[5] / f,
            0, 0, 1
        };
        transform.translation = {H[2], H[5], scale * f};
        
        result.extrinsics.push_back(transform);
    }
    
    // Calculate reprojection error
    float total_error = 0;
    int total_points = 0;
    
    for (size_t img = 0; img < object_points.size(); ++img) {
        const auto& H = homographies[img];
        const auto& obj_pts = object_points[img];
        const auto& img_pts = image_points[img];
        
        for (size_t i = 0; i < obj_pts.size(); ++i) {
            float x = H[0] * obj_pts[i].x + H[1] * obj_pts[i].y + H[2];
            float y = H[3] * obj_pts[i].x + H[4] * obj_pts[i].y + H[5];
            float w = H[6] * obj_pts[i].x + H[7] * obj_pts[i].y + H[8];
            
            x /= w;
            y /= w;
            
            float dx = x - img_pts[i].x;
            float dy = y - img_pts[i].y;
            total_error += dx*dx + dy*dy;
            total_points++;
        }
    }
    
    result.rms_error = std::sqrt(total_error / total_points);
    result.success = true;
    
    return result;
}

// ============================================================================
// Distortion Correction
// ============================================================================

/**
 * @brief Apply distortion to a point
 */
inline Point2f distort_point(const Point2f& point, const CameraMatrix& camera,
                              const DistortionCoeffs& dist) {
    // Normalize point
    float x = (point.x - camera.cx) / camera.fx;
    float y = (point.y - camera.cy) / camera.fy;
    
    float r2 = x*x + y*y;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    
    // Radial distortion
    float radial = 1 + dist.k1 * r2 + dist.k2 * r4 + dist.k3 * r6;
    
    // Tangential distortion
    float x_tang = 2 * dist.p1 * x * y + dist.p2 * (r2 + 2 * x * x);
    float y_tang = dist.p1 * (r2 + 2 * y * y) + 2 * dist.p2 * x * y;
    
    float x_dist = x * radial + x_tang;
    float y_dist = y * radial + y_tang;
    
    // Denormalize
    return Point2f(x_dist * camera.fx + camera.cx,
                   y_dist * camera.fy + camera.cy);
}

/**
 * @brief Remove distortion from a point
 */
inline Point2f undistort_point(const Point2f& point, const CameraMatrix& camera,
                                const DistortionCoeffs& dist, int max_iter = 10) {
    // Normalize point
    float x = (point.x - camera.cx) / camera.fx;
    float y = (point.y - camera.cy) / camera.fy;
    
    // Iteratively find undistorted point
    float x0 = x, y0 = y;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        float r2 = x0*x0 + y0*y0;
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        
        float radial = 1 + dist.k1 * r2 + dist.k2 * r4 + dist.k3 * r6;
        
        float x_tang = 2 * dist.p1 * x0 * y0 + dist.p2 * (r2 + 2 * x0 * x0);
        float y_tang = dist.p1 * (r2 + 2 * y0 * y0) + 2 * dist.p2 * x0 * y0;
        
        x0 = (x - x_tang) / radial;
        y0 = (y - y_tang) / radial;
    }
    
    return Point2f(x0 * camera.fx + camera.cx,
                   y0 * camera.fy + camera.cy);
}

/**
 * @brief Undistort an entire image
 */
inline Image undistort(const Image& image, const CameraMatrix& camera,
                        const DistortionCoeffs& dist) {
    Image result(image.width(), image.height(), image.channels());
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            Point2f distorted = distort_point(Point2f(static_cast<float>(x), static_cast<float>(y)),
                                              camera, dist);
            
            int sx = static_cast<int>(distorted.x);
            int sy = static_cast<int>(distorted.y);
            
            if (sx >= 0 && sx < image.width() && sy >= 0 && sy < image.height()) {
                for (int c = 0; c < image.channels(); ++c) {
                    result.at(x, y, c) = image.at(sx, sy, c);
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief Initialize undistortion maps for efficient remapping
 */
inline std::pair<Image, Image> init_undistort_rectify_map(
    const CameraMatrix& camera, const DistortionCoeffs& dist,
    int width, int height, const CameraMatrix* new_camera = nullptr) {
    
    Image map_x(width, height, 1);
    Image map_y(width, height, 1);
    
    CameraMatrix cam = new_camera ? *new_camera : camera;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Point2f distorted = distort_point(Point2f(static_cast<float>(x), static_cast<float>(y)),
                                              cam, dist);
            map_x.at(x, y, 0) = distorted.x;
            map_y.at(x, y, 0) = distorted.y;
        }
    }
    
    return {map_x, map_y};
}

/**
 * @brief Remap image using precomputed maps
 */
inline Image remap(const Image& image, const Image& map_x, const Image& map_y) {
    Image result(map_x.width(), map_x.height(), image.channels());
    
    for (int y = 0; y < result.height(); ++y) {
        for (int x = 0; x < result.width(); ++x) {
            float sx = map_x.at(x, y, 0);
            float sy = map_y.at(x, y, 0);
            
            // Bilinear interpolation
            int x0 = static_cast<int>(sx);
            int y0 = static_cast<int>(sy);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            if (x0 >= 0 && x1 < image.width() && y0 >= 0 && y1 < image.height()) {
                float fx = sx - x0;
                float fy = sy - y0;
                
                for (int c = 0; c < image.channels(); ++c) {
                    float v00 = image.at(x0, y0, c);
                    float v10 = image.at(x1, y0, c);
                    float v01 = image.at(x0, y1, c);
                    float v11 = image.at(x1, y1, c);
                    
                    float val = (1-fx) * (1-fy) * v00 + fx * (1-fy) * v10 +
                               (1-fx) * fy * v01 + fx * fy * v11;
                    result.at(x, y, c) = val;
                }
            }
        }
    }
    
    return result;
}

// ============================================================================
// Pose Estimation
// ============================================================================

/**
 * @brief Solve PnP (Perspective-n-Point) problem
 * 
 * Estimate camera pose from 3D-2D point correspondences
 */
inline bool solve_pnp(const std::vector<Point3f>& object_points,
                      const std::vector<Point2f>& image_points,
                      const CameraMatrix& camera,
                      const DistortionCoeffs& dist,
                      RigidTransform& transform) {
    if (object_points.size() < 4 || object_points.size() != image_points.size()) {
        return false;
    }
    
    // Undistort image points
    std::vector<Point2f> undistorted_pts;
    for (const auto& pt : image_points) {
        undistorted_pts.push_back(undistort_point(pt, camera, dist));
    }
    
    // Compute centroid of object points
    float cx = 0, cy = 0, cz = 0;
    for (const auto& pt : object_points) {
        cx += pt.x;
        cy += pt.y;
        cz += pt.z;
    }
    cx /= object_points.size();
    cy /= object_points.size();
    cz /= object_points.size();
    
    // Compute centroid of image points
    float ux = 0, uy = 0;
    for (const auto& pt : undistorted_pts) {
        ux += (pt.x - camera.cx) / camera.fx;
        uy += (pt.y - camera.cy) / camera.fy;
    }
    ux /= undistorted_pts.size();
    uy /= undistorted_pts.size();
    
    // Estimate depth (simplified)
    float sum_d = 0;
    for (size_t i = 0; i < object_points.size(); ++i) {
        float ox = object_points[i].x - cx;
        float oy = object_points[i].y - cy;
        float od = std::sqrt(ox*ox + oy*oy);
        
        float ix = (undistorted_pts[i].x - camera.cx) / camera.fx - ux;
        float iy = (undistorted_pts[i].y - camera.cy) / camera.fy - uy;
        float id = std::sqrt(ix*ix + iy*iy);
        
        if (id > 1e-6f) {
            sum_d += od / id;
        }
    }
    float depth = sum_d / object_points.size();
    
    // Estimate translation
    transform.translation[0] = ux * depth - cx;
    transform.translation[1] = uy * depth - cy;
    transform.translation[2] = depth;
    
    // Identity rotation (simplified - full solution would use SVD)
    transform.rotation = {1,0,0, 0,1,0, 0,0,1};
    
    return true;
}

/**
 * @brief Project 3D points to image plane
 */
inline std::vector<Point2f> project_points(const std::vector<Point3f>& object_points,
                                            const RigidTransform& transform,
                                            const CameraMatrix& camera,
                                            const DistortionCoeffs& dist) {
    std::vector<Point2f> image_points;
    
    for (const auto& pt : object_points) {
        // Apply rotation
        float x = transform.rotation[0] * pt.x + transform.rotation[1] * pt.y + transform.rotation[2] * pt.z;
        float y = transform.rotation[3] * pt.x + transform.rotation[4] * pt.y + transform.rotation[5] * pt.z;
        float z = transform.rotation[6] * pt.x + transform.rotation[7] * pt.y + transform.rotation[8] * pt.z;
        
        // Apply translation
        x += transform.translation[0];
        y += transform.translation[1];
        z += transform.translation[2];
        
        // Project to image plane
        if (z > 0) {
            float px = x / z;
            float py = y / z;
            
            // Apply camera matrix
            float u = camera.fx * px + camera.cx;
            float v = camera.fy * py + camera.cy;
            
            // Apply distortion
            Point2f distorted = distort_point(Point2f(u, v), camera, dist);
            image_points.push_back(distorted);
        } else {
            image_points.push_back(Point2f(-1, -1));  // Behind camera
        }
    }
    
    return image_points;
}

// ============================================================================
// Stereo Calibration
// ============================================================================

/**
 * @brief Stereo camera calibration result
 */
struct StereoCalibrationResult {
    CalibrationResult left;
    CalibrationResult right;
    RigidTransform stereo_transform;  // Right camera relative to left
    float rms_error = 0;
    bool success = false;
};

/**
 * @brief Compute fundamental matrix from point correspondences
 */
inline std::array<float, 9> compute_fundamental_matrix(
    const std::vector<Point2f>& points1,
    const std::vector<Point2f>& points2) {
    
    std::array<float, 9> F = {0,0,0, 0,0,0, 0,0,1};
    
    if (points1.size() < 8 || points1.size() != points2.size()) {
        return F;
    }
    
    // Normalize points
    float mean1_x = 0, mean1_y = 0, mean2_x = 0, mean2_y = 0;
    for (size_t i = 0; i < points1.size(); ++i) {
        mean1_x += points1[i].x;
        mean1_y += points1[i].y;
        mean2_x += points2[i].x;
        mean2_y += points2[i].y;
    }
    mean1_x /= points1.size();
    mean1_y /= points1.size();
    mean2_x /= points2.size();
    mean2_y /= points2.size();
    
    // Simplified fundamental matrix estimation
    // Full implementation would use 8-point algorithm with RANSAC
    
    float sum_xx = 0, sum_xy = 0, sum_yx = 0, sum_yy = 0;
    for (size_t i = 0; i < points1.size(); ++i) {
        float x1 = points1[i].x - mean1_x;
        float y1 = points1[i].y - mean1_y;
        float x2 = points2[i].x - mean2_x;
        float y2 = points2[i].y - mean2_y;
        
        sum_xx += x1 * x2;
        sum_xy += x1 * y2;
        sum_yx += y1 * x2;
        sum_yy += y1 * y2;
    }
    
    // Simplified F matrix (would need SVD for proper solution)
    F[0] = 0; F[1] = -1; F[2] = mean2_y - mean1_y;
    F[3] = 1; F[4] = 0; F[5] = mean1_x - mean2_x;
    F[6] = mean1_y - mean2_y; F[7] = mean2_x - mean1_x; F[8] = 0;
    
    return F;
}

/**
 * @brief Compute epipolar lines
 */
inline void compute_epilines(const std::vector<Point2f>& points,
                              const std::array<float, 9>& F,
                              std::vector<std::array<float, 3>>& lines,
                              bool from_image1 = true) {
    lines.clear();
    
    for (const auto& pt : points) {
        std::array<float, 3> line;
        if (from_image1) {
            // F * p1 = l2
            line[0] = F[0] * pt.x + F[1] * pt.y + F[2];
            line[1] = F[3] * pt.x + F[4] * pt.y + F[5];
            line[2] = F[6] * pt.x + F[7] * pt.y + F[8];
        } else {
            // F^T * p2 = l1
            line[0] = F[0] * pt.x + F[3] * pt.y + F[6];
            line[1] = F[1] * pt.x + F[4] * pt.y + F[7];
            line[2] = F[2] * pt.x + F[5] * pt.y + F[8];
        }
        
        // Normalize line
        float norm = std::sqrt(line[0]*line[0] + line[1]*line[1]);
        if (norm > 1e-6f) {
            line[0] /= norm;
            line[1] /= norm;
            line[2] /= norm;
        }
        
        lines.push_back(line);
    }
}

} // namespace calibration
} // namespace neurova
