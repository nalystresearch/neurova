// Copyright (c) 2026 Neurova - FINAL MODULE
// Stitching, Timeseries, Transform, and Utils in C++
// ~2,400 lines Python -> Optimized C++

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <memory>
#include <limits>
#include <tuple>

#ifdef __ARM_NEON
#include <arm_neon.h>
#define SIMD_TYPE "ARM NEON"
#elif defined(__AVX2__)
#include <immintrin.h>
#define SIMD_TYPE "AVX2"
#else
#define SIMD_TYPE "None"
#endif

namespace py = pybind11;

// =============================================================================
// UTILS MODULE - Drawing Utilities
// =============================================================================

using Color = std::tuple<int, int, int>;
using Point = std::pair<int, int>;

py::array_t<uint8_t> draw_rectangle(
    py::array_t<uint8_t> image,
    Point pt1,
    Point pt2,
    Color color = {0, 255, 0},
    int thickness = 1
) {
    auto buf = image.request();
    int h = buf.shape[0];
    int w = buf.shape[1];
    int c = (buf.ndim == 3) ? buf.shape[2] : 1;
    
    auto ptr = static_cast<uint8_t*>(buf.ptr);
    std::vector<uint8_t> result(ptr, ptr + buf.size);
    
    int x1 = std::get<0>(pt1);
    int y1 = std::get<1>(pt1);
    int x2 = std::get<0>(pt2);
    int y2 = std::get<1>(pt2);
    
    int x_min = std::min(x1, x2);
    int x_max = std::max(x1, x2);
    int y_min = std::min(y1, y2);
    int y_max = std::max(y1, y2);
    
    auto [r, g, b] = color;
    
    // Draw rectangle edges
    for (int t = 0; t < thickness; ++t) {
        // Top and bottom
        for (int x = x_min; x <= x_max; ++x) {
            if (x >= 0 && x < w) {
                // Top
                int y = y_min + t;
                if (y >= 0 && y < h && c >= 3) {
                    int idx = (y * w + x) * c;
                    result[idx] = r;
                    result[idx + 1] = g;
                    result[idx + 2] = b;
                }
                // Bottom
                y = y_max - t;
                if (y >= 0 && y < h && c >= 3) {
                    int idx = (y * w + x) * c;
                    result[idx] = r;
                    result[idx + 1] = g;
                    result[idx + 2] = b;
                }
            }
        }
        
        // Left and right
        for (int y = y_min; y <= y_max; ++y) {
            if (y >= 0 && y < h) {
                // Left
                int x = x_min + t;
                if (x >= 0 && x < w && c >= 3) {
                    int idx = (y * w + x) * c;
                    result[idx] = r;
                    result[idx + 1] = g;
                    result[idx + 2] = b;
                }
                // Right
                x = x_max - t;
                if (x >= 0 && x < w && c >= 3) {
                    int idx = (y * w + x) * c;
                    result[idx] = r;
                    result[idx + 1] = g;
                    result[idx + 2] = b;
                }
            }
        }
    }
    
    return py::array_t<uint8_t>(buf.shape, result.data());
}

py::array_t<uint8_t> draw_circle(
    py::array_t<uint8_t> image,
    Point center,
    int radius,
    Color color = {0, 255, 0},
    int thickness = 1
) {
    auto buf = image.request();
    int h = buf.shape[0];
    int w = buf.shape[1];
    int c = (buf.ndim == 3) ? buf.shape[2] : 1;
    
    auto ptr = static_cast<uint8_t*>(buf.ptr);
    std::vector<uint8_t> result(ptr, ptr + buf.size);
    
    int cx = std::get<0>(center);
    int cy = std::get<1>(center);
    auto [r, g, b] = color;
    
    // Midpoint circle algorithm
    int x = 0;
    int y = radius;
    int d = 3 - 2 * radius;
    
    auto draw_pixel = [&](int px, int py) {
        if (px >= 0 && px < w && py >= 0 && py < h && c >= 3) {
            int idx = (py * w + px) * c;
            result[idx] = r;
            result[idx + 1] = g;
            result[idx + 2] = b;
        }
    };
    
    auto draw_circle_points = [&](int x, int y) {
        for (int t = 0; t < thickness; ++t) {
            draw_pixel(cx + x, cy + y - t);
            draw_pixel(cx - x, cy + y - t);
            draw_pixel(cx + x, cy - y + t);
            draw_pixel(cx - x, cy - y + t);
            draw_pixel(cx + y, cy + x - t);
            draw_pixel(cx - y, cy + x - t);
            draw_pixel(cx + y, cy - x + t);
            draw_pixel(cx - y, cy - x + t);
        }
    };
    
    while (y >= x) {
        draw_circle_points(x, y);
        x++;
        if (d > 0) {
            y--;
            d = d + 4 * (x - y) + 10;
        } else {
            d = d + 4 * x + 6;
        }
    }
    
    return py::array_t<uint8_t>(buf.shape, result.data());
}

py::array_t<uint8_t> draw_line(
    py::array_t<uint8_t> image,
    Point pt1,
    Point pt2,
    Color color = {0, 255, 0},
    int thickness = 1
) {
    auto buf = image.request();
    int h = buf.shape[0];
    int w = buf.shape[1];
    int c = (buf.ndim == 3) ? buf.shape[2] : 1;
    
    auto ptr = static_cast<uint8_t*>(buf.ptr);
    std::vector<uint8_t> result(ptr, ptr + buf.size);
    
    int x0 = std::get<0>(pt1);
    int y0 = std::get<1>(pt1);
    int x1 = std::get<0>(pt2);
    int y1 = std::get<1>(pt2);
    
    auto [r, g, b] = color;
    
    // Bresenham's line algorithm
    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    
    auto draw_point = [&](int x, int y) {
        for (int t = 0; t < thickness; ++t) {
            for (int tx = -t; tx <= t; ++tx) {
                for (int ty = -t; ty <= t; ++ty) {
                    int px = x + tx;
                    int py = y + ty;
                    if (px >= 0 && px < w && py >= 0 && py < h && c >= 3) {
                        int idx = (py * w + px) * c;
                        result[idx] = r;
                        result[idx + 1] = g;
                        result[idx + 2] = b;
                    }
                }
            }
        }
    };
    
    while (true) {
        draw_point(x0, y0);
        if (x0 == x1 && y0 == y1) break;
        
        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
    
    return py::array_t<uint8_t>(buf.shape, result.data());
}

// =============================================================================
// TRANSFORM MODULE - Geometric Transformations
// =============================================================================

py::array_t<double> get_rotation_matrix2d(
    std::pair<double, double> center,
    double angle_degrees,
    double scale = 1.0
) {
    double cx = center.first;
    double cy = center.second;
    double a = angle_degrees * M_PI / 180.0;
    double alpha = scale * std::cos(a);
    double beta = scale * std::sin(a);
    
    std::vector<double> m = {
        alpha, beta, (1.0 - alpha) * cx - beta * cy,
        -beta, alpha, beta * cx + (1.0 - alpha) * cy
    };
    
    return py::array_t<double>({2, 3}, m.data());
}

py::array_t<uint8_t> warp_affine(
    py::array_t<uint8_t> src,
    py::array_t<double> M,
    std::pair<int, int> dsize
) {
    auto src_buf = src.request();
    auto M_buf = M.request();
    
    int src_h = src_buf.shape[0];
    int src_w = src_buf.shape[1];
    int c = (src_buf.ndim == 3) ? src_buf.shape[2] : 1;
    
    int dst_w = dsize.first;
    int dst_h = dsize.second;
    
    auto src_ptr = static_cast<uint8_t*>(src_buf.ptr);
    auto M_ptr = static_cast<double*>(M_buf.ptr);
    
    std::vector<uint8_t> dst(dst_h * dst_w * c, 0);
    
    // Inverse transformation
    double m00 = M_ptr[0], m01 = M_ptr[1], m02 = M_ptr[2];
    double m10 = M_ptr[3], m11 = M_ptr[4], m12 = M_ptr[5];
    
    // Compute inverse
    double det = m00 * m11 - m01 * m10;
    if (std::abs(det) < 1e-10) {
        return py::array_t<uint8_t>({dst_h, dst_w, c}, dst.data());
    }
    
    double inv00 = m11 / det;
    double inv01 = -m01 / det;
    double inv10 = -m10 / det;
    double inv11 = m00 / det;
    
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            // Map destination to source
            double sx = inv00 * (x - m02) + inv01 * (y - m12);
            double sy = inv10 * (x - m02) + inv11 * (y - m12);
            
            int ix = static_cast<int>(sx);
            int iy = static_cast<int>(sy);
            
            if (ix >= 0 && ix < src_w && iy >= 0 && iy < src_h) {
                for (int ch = 0; ch < c; ++ch) {
                    dst[(y * dst_w + x) * c + ch] = src_ptr[(iy * src_w + ix) * c + ch];
                }
            }
        }
    }
    
    std::vector<ssize_t> shape = {dst_h, dst_w};
    if (c > 1) shape.push_back(c);
    
    return py::array_t<uint8_t>(shape, dst.data());
}

py::array_t<uint8_t> resize(
    py::array_t<uint8_t> src,
    std::pair<int, int> dsize
) {
    auto buf = src.request();
    int src_h = buf.shape[0];
    int src_w = buf.shape[1];
    int c = (buf.ndim == 3) ? buf.shape[2] : 1;
    
    int dst_w = dsize.first;
    int dst_h = dsize.second;
    
    auto src_ptr = static_cast<uint8_t*>(buf.ptr);
    std::vector<uint8_t> dst(dst_h * dst_w * c);
    
    float x_ratio = static_cast<float>(src_w) / dst_w;
    float y_ratio = static_cast<float>(src_h) / dst_h;
    
    // Nearest neighbor interpolation
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            int src_x = static_cast<int>(x * x_ratio);
            int src_y = static_cast<int>(y * y_ratio);
            
            for (int ch = 0; ch < c; ++ch) {
                dst[(y * dst_w + x) * c + ch] = src_ptr[(src_y * src_w + src_x) * c + ch];
            }
        }
    }
    
    std::vector<ssize_t> shape = {dst_h, dst_w};
    if (c > 1) shape.push_back(c);
    
    return py::array_t<uint8_t>(shape, dst.data());
}

py::array_t<uint8_t> rotate(
    py::array_t<uint8_t> src,
    double angle_degrees
) {
    auto buf = src.request();
    int h = buf.shape[0];
    int w = buf.shape[1];
    int c = (buf.ndim == 3) ? buf.shape[2] : 1;
    
    // Get rotation matrix
    auto M = get_rotation_matrix2d({w/2.0, h/2.0}, angle_degrees, 1.0);
    
    // Apply warp
    return warp_affine(src, M, {w, h});
}

// =============================================================================
// TIMESERIES MODULE - Time Series Analysis
// =============================================================================

py::array_t<double> difference(py::array_t<double> y, int d = 1) {
    auto buf = y.request();
    auto ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.size;
    
    std::vector<double> result(ptr, ptr + n);
    
    for (int iter = 0; iter < d; ++iter) {
        std::vector<double> diff;
        for (size_t i = 1; i < result.size(); ++i) {
            diff.push_back(result[i] - result[i-1]);
        }
        result = diff;
    }
    
    return py::array_t<double>(result.size(), result.data());
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> 
seasonal_decompose(
    py::array_t<double> y,
    int period = 12
) {
    auto buf = y.request();
    auto ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.size;
    
    std::vector<double> trend(n);
    std::vector<double> seasonal(n);
    std::vector<double> residual(n);
    
    // Simple moving average for trend
    int window = period;
    for (size_t i = 0; i < n; ++i) {
        if (i < window/2 || i >= n - window/2) {
            trend[i] = ptr[i];
        } else {
            double sum = 0;
            for (int j = -window/2; j <= window/2; ++j) {
                sum += ptr[i + j];
            }
            trend[i] = sum / window;
        }
    }
    
    // Detrend to get seasonal + residual
    std::vector<double> detrended(n);
    for (size_t i = 0; i < n; ++i) {
        detrended[i] = ptr[i] - trend[i];
    }
    
    // Average by season
    std::vector<double> seasonal_avg(period, 0.0);
    std::vector<int> seasonal_count(period, 0);
    
    for (size_t i = 0; i < n; ++i) {
        seasonal_avg[i % period] += detrended[i];
        seasonal_count[i % period]++;
    }
    
    for (int j = 0; j < period; ++j) {
        if (seasonal_count[j] > 0) {
            seasonal_avg[j] /= seasonal_count[j];
        }
    }
    
    // Assign seasonal component
    for (size_t i = 0; i < n; ++i) {
        seasonal[i] = seasonal_avg[i % period];
        residual[i] = ptr[i] - trend[i] - seasonal[i];
    }
    
    return {
        py::array_t<double>(n, trend.data()),
        py::array_t<double>(n, seasonal.data()),
        py::array_t<double>(n, residual.data())
    };
}

py::array_t<double> exponential_smoothing(
    py::array_t<double> y,
    double alpha = 0.3
) {
    auto buf = y.request();
    auto ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.size;
    
    std::vector<double> smoothed(n);
    smoothed[0] = ptr[0];
    
    for (size_t i = 1; i < n; ++i) {
        smoothed[i] = alpha * ptr[i] + (1.0 - alpha) * smoothed[i-1];
    }
    
    return py::array_t<double>(n, smoothed.data());
}

double acf_at_lag(py::array_t<double> y, int lag) {
    auto buf = y.request();
    auto ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.size;
    
    // Compute mean
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mean += ptr[i];
    }
    mean /= n;
    
    // Compute autocovariance at lag
    double auto_cov = 0.0;
    double variance = 0.0;
    
    for (size_t i = 0; i < n - lag; ++i) {
        auto_cov += (ptr[i] - mean) * (ptr[i + lag] - mean);
    }
    
    for (size_t i = 0; i < n; ++i) {
        variance += (ptr[i] - mean) * (ptr[i] - mean);
    }
    
    return (variance > 0) ? (auto_cov / variance) : 0.0;
}

py::array_t<double> acf(py::array_t<double> y, int nlags = 40) {
    std::vector<double> acf_values(nlags + 1);
    
    for (int lag = 0; lag <= nlags; ++lag) {
        acf_values[lag] = acf_at_lag(y, lag);
    }
    
    return py::array_t<double>(nlags + 1, acf_values.data());
}

// =============================================================================
// STITCHING MODULE - Image Stitching
// =============================================================================

enum class StitcherMode {
    PANORAMA = 0,
    SCANS = 1
};

enum class StitcherStatus {
    OK = 0,
    ERR_NEED_MORE_IMGS = 1,
    ERR_HOMOGRAPHY_EST_FAIL = 2,
    ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
};

class Stitcher {
private:
    StitcherMode mode_;
    double registration_resol_;
    double seam_estimation_resol_;
    double compositing_resol_;
    
public:
    Stitcher(StitcherMode mode = StitcherMode::PANORAMA)
        : mode_(mode),
          registration_resol_(0.6),
          seam_estimation_resol_(0.1),
          compositing_resol_(-1.0) {}
    
    static Stitcher create(StitcherMode mode = StitcherMode::PANORAMA) {
        return Stitcher(mode);
    }
    
    std::pair<StitcherStatus, py::array_t<uint8_t>> stitch(
        std::vector<py::array_t<uint8_t>> images
    ) {
        if (images.size() < 2) {
            std::vector<uint8_t> empty(3, 0);
            return {StitcherStatus::ERR_NEED_MORE_IMGS, 
                    py::array_t<uint8_t>({1, 1, 3}, empty.data())};
        }
        
        // Simple horizontal stitching (simplified implementation)
        auto first_buf = images[0].request();
        int h = first_buf.shape[0];
        int w = first_buf.shape[1];
        int c = first_buf.shape[2];
        
        // Calculate total width
        int total_w = 0;
        for (const auto& img : images) {
            auto buf = img.request();
            total_w += buf.shape[1];
        }
        
        // Create result image
        std::vector<uint8_t> result(h * total_w * c, 0);
        
        // Copy images side by side
        int x_offset = 0;
        for (const auto& img : images) {
            auto buf = img.request();
            auto ptr = static_cast<uint8_t*>(buf.ptr);
            int img_w = buf.shape[1];
            
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < img_w; ++x) {
                    for (int ch = 0; ch < c; ++ch) {
                        result[(y * total_w + (x_offset + x)) * c + ch] = 
                            ptr[(y * img_w + x) * c + ch];
                    }
                }
            }
            x_offset += img_w;
        }
        
        return {StitcherStatus::OK, 
                py::array_t<uint8_t>({h, total_w, c}, result.data())};
    }
    
    StitcherMode get_mode() const { return mode_; }
};

// =============================================================================
// PYBIND11 BINDINGS
// =============================================================================

PYBIND11_MODULE(neurova_final, m) {
    m.doc() = "Neurova Final Module - Stitching, Timeseries, Transform, Utils";
    m.attr("__version__") = "1.0.0";
    m.attr("SIMD") = SIMD_TYPE;
    
    // =========================================================================
    // UTILS MODULE
    // =========================================================================
    
    py::module_ utils = m.def_submodule("utils", "Drawing utilities");
    
    utils.def("draw_rectangle", &draw_rectangle,
             "Draw rectangle on image",
             py::arg("image"), py::arg("pt1"), py::arg("pt2"),
             py::arg("color") = Color{0, 255, 0},
             py::arg("thickness") = 1);
    
    utils.def("draw_circle", &draw_circle,
             "Draw circle on image",
             py::arg("image"), py::arg("center"), py::arg("radius"),
             py::arg("color") = Color{0, 255, 0},
             py::arg("thickness") = 1);
    
    utils.def("draw_line", &draw_line,
             "Draw line on image",
             py::arg("image"), py::arg("pt1"), py::arg("pt2"),
             py::arg("color") = Color{0, 255, 0},
             py::arg("thickness") = 1);
    
    // =========================================================================
    // TRANSFORM MODULE
    // =========================================================================
    
    py::module_ transform = m.def_submodule("transform", "Geometric transforms");
    
    transform.def("get_rotation_matrix2d", &get_rotation_matrix2d,
                 "Get 2D rotation matrix",
                 py::arg("center"), py::arg("angle_degrees"),
                 py::arg("scale") = 1.0);
    
    transform.def("warp_affine", &warp_affine,
                 "Apply affine transformation",
                 py::arg("src"), py::arg("M"), py::arg("dsize"));
    
    transform.def("resize", &resize,
                 "Resize image",
                 py::arg("src"), py::arg("dsize"));
    
    transform.def("rotate", &rotate,
                 "Rotate image",
                 py::arg("src"), py::arg("angle_degrees"));
    
    // =========================================================================
    // TIMESERIES MODULE
    // =========================================================================
    
    py::module_ ts = m.def_submodule("timeseries", "Time series analysis");
    
    ts.def("difference", &difference,
          "Apply differencing to time series",
          py::arg("y"), py::arg("d") = 1);
    
    ts.def("seasonal_decompose", &seasonal_decompose,
          "Decompose time series into trend, seasonal, residual",
          py::arg("y"), py::arg("period") = 12);
    
    ts.def("exponential_smoothing", &exponential_smoothing,
          "Apply exponential smoothing",
          py::arg("y"), py::arg("alpha") = 0.3);
    
    ts.def("acf", &acf,
          "Compute autocorrelation function",
          py::arg("y"), py::arg("nlags") = 40);
    
    ts.def("acf_at_lag", &acf_at_lag,
          "Compute ACF at specific lag",
          py::arg("y"), py::arg("lag"));
    
    // =========================================================================
    // STITCHING MODULE
    // =========================================================================
    
    py::module_ stitch = m.def_submodule("stitching", "Image stitching");
    
    py::enum_<StitcherMode>(stitch, "Mode")
        .value("PANORAMA", StitcherMode::PANORAMA)
        .value("SCANS", StitcherMode::SCANS)
        .export_values();
    
    py::enum_<StitcherStatus>(stitch, "Status")
        .value("OK", StitcherStatus::OK)
        .value("ERR_NEED_MORE_IMGS", StitcherStatus::ERR_NEED_MORE_IMGS)
        .value("ERR_HOMOGRAPHY_EST_FAIL", StitcherStatus::ERR_HOMOGRAPHY_EST_FAIL)
        .value("ERR_CAMERA_PARAMS_ADJUST_FAIL", StitcherStatus::ERR_CAMERA_PARAMS_ADJUST_FAIL)
        .export_values();
    
    py::class_<Stitcher>(stitch, "Stitcher")
        .def(py::init<StitcherMode>(), py::arg("mode") = StitcherMode::PANORAMA)
        .def_static("create", &Stitcher::create,
                   py::arg("mode") = StitcherMode::PANORAMA)
        .def("stitch", &Stitcher::stitch, py::arg("images"))
        .def("get_mode", &Stitcher::get_mode);
}
