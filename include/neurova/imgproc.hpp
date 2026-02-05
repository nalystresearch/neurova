// Copyright (c) 2026 @squid consultancy group (scg)
// all rights reserved.
// licensed under the apache license 2.0.

/**
 * neurova/imgproc.hpp - Image Processing Operations
 * 
 * This header provides all image processing functions:
 * - Color conversion (RGB, BGR, HSV, Grayscale, etc.)
 * - Filters (Gaussian, Sobel, Laplacian, Bilateral, etc.)
 * - Geometric transforms (resize, rotate, warp, etc.)
 * - Morphological operations (erosion, dilation, etc.)
 * - Histogram operations
 * - Edge detection
 */

#ifndef NEUROVA_IMGPROC_HPP
#define NEUROVA_IMGPROC_HPP

#include "core.hpp"

namespace neurova {
namespace imgproc {

// ============================================================================
// Color Conversion
// ============================================================================

enum class ColorCode {
    BGR2RGB,
    RGB2BGR,
    BGR2GRAY,
    RGB2GRAY,
    GRAY2BGR,
    GRAY2RGB,
    BGR2HSV,
    RGB2HSV,
    HSV2BGR,
    HSV2RGB,
    BGR2LAB,
    RGB2LAB,
    LAB2BGR,
    LAB2RGB,
    BGR2YUV,
    RGB2YUV,
    YUV2BGR,
    YUV2RGB,
    BGR2YCrCb,
    RGB2YCrCb,
    YCrCb2BGR,
    YCrCb2RGB
};

Image cvtColor(const Image& src, ColorCode code);
Image bgr2rgb(const Image& src);
Image rgb2bgr(const Image& src);
Image bgr2gray(const Image& src);
Image rgb2gray(const Image& src);
Image gray2bgr(const Image& src);
Image gray2rgb(const Image& src);
Image bgr2hsv(const Image& src);
Image rgb2hsv(const Image& src);
Image hsv2bgr(const Image& src);
Image hsv2rgb(const Image& src);

// ============================================================================
// Filtering
// ============================================================================

// blur filters
Image blur(const Image& src, int ksize);
Image gaussianBlur(const Image& src, int ksize, double sigma = 0.0);
Image medianBlur(const Image& src, int ksize);
Image bilateralFilter(const Image& src, int d, double sigmaColor, double sigmaSpace);
Image boxFilter(const Image& src, int ksize, bool normalize = true);

// custom filter
Image filter2D(const Image& src, const Tensor& kernel);
Tensor convolve2D(const Tensor& src, const Tensor& kernel, bool same = true);

// edge detection
Image sobel(const Image& src, int dx, int dy, int ksize = 3);
Image scharr(const Image& src, int dx, int dy);
Image laplacian(const Image& src, int ksize = 3);
Image canny(const Image& src, double threshold1, double threshold2, int apertureSize = 3);

// gradient
std::pair<Image, Image> spatialGradient(const Image& src);
Image magnitude(const Image& gx, const Image& gy);
Image phase(const Image& gx, const Image& gy, bool angleInDegrees = false);

// sharpening
Image sharpen(const Image& src, double amount = 1.0);
Image unsharpMask(const Image& src, int ksize, double sigma, double amount);

// noise reduction
Image fastNlMeansDenoising(const Image& src, float h = 3.0f, int templateWindowSize = 7, int searchWindowSize = 21);

// ============================================================================
// Geometric Transforms
// ============================================================================

enum class InterpolationMode {
    NEAREST,
    LINEAR,
    BILINEAR = LINEAR,  // alias
    CUBIC,
    BICUBIC = CUBIC,    // alias
    AREA,
    LANCZOS4,
    LANCZOS = LANCZOS4  // alias
};

enum class BorderMode {
    CONSTANT,
    REPLICATE,
    REFLECT,
    WRAP,
    REFLECT_101
};

// resize
Image resize(const Image& src, size_t width, size_t height, InterpolationMode interp = InterpolationMode::LINEAR);
Image resize(const Image& src, double fx, double fy, InterpolationMode interp = InterpolationMode::LINEAR);

// rotation
Image rotate(const Image& src, double angle, bool expand = false);
Image rotate90(const Image& src, int k = 1);

// flip
Image flip(const Image& src, int flipCode);
Image flipHorizontal(const Image& src);
Image flipVertical(const Image& src);

// affine transforms
Tensor getRotationMatrix2D(double cx, double cy, double angle, double scale);
Tensor getAffineTransform(const std::vector<std::pair<double, double>>& src,
                          const std::vector<std::pair<double, double>>& dst);
Image warpAffine(const Image& src, const Tensor& M, size_t width, size_t height,
                 InterpolationMode interp = InterpolationMode::LINEAR,
                 BorderMode border = BorderMode::CONSTANT);

// perspective transforms
Tensor getPerspectiveTransform(const std::vector<std::pair<double, double>>& src,
                               const std::vector<std::pair<double, double>>& dst);
Image warpPerspective(const Image& src, const Tensor& M, size_t width, size_t height,
                      InterpolationMode interp = InterpolationMode::LINEAR,
                      BorderMode border = BorderMode::CONSTANT);

// crop
Image crop(const Image& src, int x, int y, int width, int height);
Image centerCrop(const Image& src, int width, int height);

// pad
Image copyMakeBorder(const Image& src, int top, int bottom, int left, int right,
                     BorderMode border = BorderMode::CONSTANT, double value = 0.0);
Image pad(const Image& src, int padding, BorderMode border = BorderMode::CONSTANT, double value = 0.0);

// ============================================================================
// Morphological Operations
// ============================================================================

enum class MorphShape {
    RECT,
    CROSS,
    ELLIPSE
};

enum class MorphOp {
    ERODE,
    DILATE,
    OPEN,
    CLOSE,
    GRADIENT,
    TOPHAT,
    BLACKHAT,
    HITMISS
};

Tensor getStructuringElement(MorphShape shape, int ksize);
Image erode(const Image& src, const Tensor& kernel, int iterations = 1);
Image dilate(const Image& src, const Tensor& kernel, int iterations = 1);
Image morphologyEx(const Image& src, MorphOp op, const Tensor& kernel, int iterations = 1);
Image opening(const Image& src, const Tensor& kernel);
Image closing(const Image& src, const Tensor& kernel);
Image morphGradient(const Image& src, const Tensor& kernel);
Image topHat(const Image& src, const Tensor& kernel);
Image blackHat(const Image& src, const Tensor& kernel);

// ============================================================================
// Histogram Operations
// ============================================================================

Tensor calcHist(const Image& src, int histSize = 256, double minVal = 0.0, double maxVal = 256.0);
Image equalizeHist(const Image& src);
Image clahe(const Image& src, double clipLimit = 2.0, int tileGridSize = 8);
Tensor compareHist(const Tensor& hist1, const Tensor& hist2, int method);

// ============================================================================
// Thresholding
// ============================================================================

enum class ThresholdType {
    BINARY,
    BINARY_INV,
    TRUNC,
    TOZERO,
    TOZERO_INV,
    OTSU,
    TRIANGLE
};

Image threshold(const Image& src, double thresh, double maxval, ThresholdType type);
Image adaptiveThreshold(const Image& src, double maxval, int adaptiveMethod,
                        ThresholdType type, int blockSize, double C);
double otsuThreshold(const Image& src);

// ============================================================================
// Image Analysis
// ============================================================================

// moments
struct Moments {
    double m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
    double mu20, mu11, mu02, mu30, mu21, mu12, mu03;
    double nu20, nu11, nu02, nu30, nu21, nu12, nu03;
};

Moments moments(const Image& src, bool binaryImage = false);
std::vector<double> huMoments(const Moments& m);

// contours
struct Contour {
    std::vector<std::pair<int, int>> points;
};

std::vector<Contour> findContours(const Image& src);
Image drawContours(const Image& src, const std::vector<Contour>& contours, int idx = -1);
double contourArea(const Contour& contour);
double arcLength(const Contour& contour, bool closed = true);
std::tuple<int, int, int, int> boundingRect(const Contour& contour);
Contour approxPolyDP(const Contour& contour, double epsilon, bool closed = true);
Contour convexHull(const Contour& contour);

// connected components
std::pair<int, Image> connectedComponents(const Image& src);
std::tuple<int, Image, Tensor, Tensor> connectedComponentsWithStats(const Image& src);

// ============================================================================
// Drawing
// ============================================================================

void line(Image& img, int x1, int y1, int x2, int y2, 
          const std::vector<uint8_t>& color, int thickness = 1);
void rectangle(Image& img, int x, int y, int width, int height,
               const std::vector<uint8_t>& color, int thickness = 1);
void circle(Image& img, int cx, int cy, int radius,
            const std::vector<uint8_t>& color, int thickness = 1);
void ellipse(Image& img, int cx, int cy, int a, int b, double angle,
             const std::vector<uint8_t>& color, int thickness = 1);
void polylines(Image& img, const std::vector<std::pair<int, int>>& pts,
               bool isClosed, const std::vector<uint8_t>& color, int thickness = 1);
void fillPoly(Image& img, const std::vector<std::pair<int, int>>& pts,
              const std::vector<uint8_t>& color);
void putText(Image& img, const std::string& text, int x, int y,
             double fontScale, const std::vector<uint8_t>& color, int thickness = 1);

// ============================================================================
// Integral Image
// ============================================================================

Tensor integral(const Image& src);
Tensor integralSquare(const Image& src);

// ============================================================================
// Distance Transform
// ============================================================================

Image distanceTransform(const Image& src);

} // namespace imgproc
} // namespace neurova

#endif // NEUROVA_IMGPROC_HPP
