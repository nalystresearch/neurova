# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova.imgproc - Image Processing Functions

Provides Neurova image processing functionality including:
- Color space conversions (cvtColor, COLOR_* constants)
- Drawing functions (line, circle, rectangle, putText, etc.)
- Contour operations (findContours, contourArea, etc.)
- Thresholding (threshold, adaptiveThreshold, inRange)
- Morphological operations
"""

from __future__ import annotations

# Import color conversion functions and constants
from neurova.imgproc.color import (
    cvtColor,
    # BGR <-> RGB
    COLOR_BGR2BGRA, COLOR_RGB2RGBA, COLOR_BGRA2BGR, COLOR_RGBA2RGB,
    COLOR_BGR2RGBA, COLOR_RGB2BGRA, COLOR_RGBA2BGR, COLOR_BGRA2RGB,
    COLOR_BGR2RGB, COLOR_RGB2BGR, COLOR_BGRA2RGBA, COLOR_RGBA2BGRA,
    # BGR <-> Gray
    COLOR_BGR2GRAY, COLOR_RGB2GRAY, COLOR_GRAY2BGR, COLOR_GRAY2RGB,
    COLOR_GRAY2BGRA, COLOR_GRAY2RGBA, COLOR_BGRA2GRAY, COLOR_RGBA2GRAY,
    # HSV
    COLOR_BGR2HSV, COLOR_RGB2HSV, COLOR_HSV2BGR, COLOR_HSV2RGB,
    COLOR_BGR2HSV_FULL, COLOR_RGB2HSV_FULL, COLOR_HSV2BGR_FULL, COLOR_HSV2RGB_FULL,
    # HLS
    COLOR_BGR2HLS, COLOR_RGB2HLS, COLOR_HLS2BGR, COLOR_HLS2RGB,
    # Lab
    COLOR_BGR2Lab, COLOR_RGB2Lab, COLOR_Lab2BGR, COLOR_Lab2RGB,
    # Luv
    COLOR_BGR2Luv, COLOR_RGB2Luv, COLOR_Luv2BGR, COLOR_Luv2RGB,
    # YCrCb
    COLOR_BGR2YCrCb, COLOR_RGB2YCrCb, COLOR_YCrCb2BGR, COLOR_YCrCb2RGB,
    # YUV
    COLOR_BGR2YUV, COLOR_RGB2YUV, COLOR_YUV2BGR, COLOR_YUV2RGB,
    # XYZ
    COLOR_BGR2XYZ, COLOR_RGB2XYZ, COLOR_XYZ2BGR, COLOR_XYZ2RGB,
)

# Import drawing functions
from neurova.imgproc.drawing import (
    line, arrowedLine, rectangle, circle, ellipse,
    polylines, fillPoly, fillConvexPoly, putText, getTextSize,
    drawMarker, drawContours,
    # Line types
    LINE_4, LINE_8, LINE_AA, FILLED,
    # Font constants
    FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX,
    FONT_HERSHEY_COMPLEX, FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL,
    FONT_HERSHEY_SCRIPT_SIMPLEX, FONT_HERSHEY_SCRIPT_COMPLEX, FONT_ITALIC,
    # Marker types
    MARKER_CROSS, MARKER_TILTED_CROSS, MARKER_STAR, MARKER_DIAMOND,
    MARKER_SQUARE, MARKER_TRIANGLE_UP, MARKER_TRIANGLE_DOWN,
)

# Import contour functions
from neurova.imgproc.contours import (
    findContours, contourArea, arcLength, boundingRect,
    minAreaRect, minEnclosingCircle, convexHull, approxPolyDP,
    moments, isContourConvex, pointPolygonTest,
    # Retrieval modes
    RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE, RETR_FLOODFILL,
    # Approximation methods
    CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS,
)

# Import threshold functions
from neurova.imgproc.threshold import (
    threshold, adaptiveThreshold, inRange,
    # Threshold types
    THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO,
    THRESH_TOZERO_INV, THRESH_MASK, THRESH_OTSU, THRESH_TRIANGLE,
    # Adaptive methods
    ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C,
)

# Import Hough transforms
from neurova.imgproc.hough import (
    HoughLines, HoughLinesP, HoughCircles,
    HOUGH_STANDARD, HOUGH_PROBABILISTIC, HOUGH_MULTI_SCALE,
    HOUGH_GRADIENT, HOUGH_GRADIENT_ALT,
)

# Import geometric transforms
from neurova.imgproc.geometric import (
    getPerspectiveTransform, getAffineTransform, getRotationMatrix2D,
    warpPerspective, warpAffine, remap, invertAffineTransform,
    perspectiveTransform, transform,
    INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4,
    INTER_LINEAR_EXACT, INTER_NEAREST_EXACT,
    WARP_FILL_OUTLIERS, WARP_INVERSE_MAP,
)

# Import histogram functions
from neurova.imgproc.histogram import (
    calcHist, compareHist, equalizeHist, calcBackProject,
    CLAHE, createCLAHE,
    HISTCMP_CORREL, HISTCMP_CHISQR, HISTCMP_INTERSECT,
    HISTCMP_BHATTACHARYYA, HISTCMP_HELLINGER, HISTCMP_CHISQR_ALT, HISTCMP_KL_DIV,
)

# Import pyramid functions
from neurova.imgproc.pyramid import (
    pyrDown, pyrUp, buildPyramid,
)

# Import connected components
from neurova.imgproc.connected import (
    connectedComponents, connectedComponentsWithStats,
    CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH, CC_STAT_HEIGHT, CC_STAT_AREA,
)

# Import template matching
from neurova.imgproc.template import (
    matchTemplate, minMaxLoc,
    TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED,
    TM_CCOEFF, TM_CCOEFF_NORMED,
)

# Import corner detection
from neurova.imgproc.corners import (
    cornerHarris, cornerMinEigenVal, cornerEigenValsAndVecs,
    cornerSubPix, preCornerDetect,
)

# Import segmentation functions
from neurova.imgproc.segmentation import (
    floodFill, distanceTransform, distanceTransformWithLabels,
    watershed, grabCut, pyrMeanShiftFiltering,
    FLOODFILL_FIXED_RANGE, FLOODFILL_MASK_ONLY,
    DIST_L1, DIST_L2, DIST_C, DIST_MASK_3, DIST_MASK_5, DIST_MASK_PRECISE,
    GC_INIT_WITH_RECT, GC_INIT_WITH_MASK, GC_EVAL, GC_EVAL_FREEZE_MODEL,
    GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD,
)


__all__ = [
    # Color conversion
    "cvtColor",
    # Color constants
    "COLOR_BGR2BGRA", "COLOR_RGB2RGBA", "COLOR_BGRA2BGR", "COLOR_RGBA2RGB",
    "COLOR_BGR2RGBA", "COLOR_RGB2BGRA", "COLOR_RGBA2BGR", "COLOR_BGRA2RGB",
    "COLOR_BGR2RGB", "COLOR_RGB2BGR", "COLOR_BGRA2RGBA", "COLOR_RGBA2BGRA",
    "COLOR_BGR2GRAY", "COLOR_RGB2GRAY", "COLOR_GRAY2BGR", "COLOR_GRAY2RGB",
    "COLOR_GRAY2BGRA", "COLOR_GRAY2RGBA", "COLOR_BGRA2GRAY", "COLOR_RGBA2GRAY",
    "COLOR_BGR2HSV", "COLOR_RGB2HSV", "COLOR_HSV2BGR", "COLOR_HSV2RGB",
    "COLOR_BGR2HSV_FULL", "COLOR_RGB2HSV_FULL", "COLOR_HSV2BGR_FULL", "COLOR_HSV2RGB_FULL",
    "COLOR_BGR2HLS", "COLOR_RGB2HLS", "COLOR_HLS2BGR", "COLOR_HLS2RGB",
    "COLOR_BGR2Lab", "COLOR_RGB2Lab", "COLOR_Lab2BGR", "COLOR_Lab2RGB",
    "COLOR_BGR2Luv", "COLOR_RGB2Luv", "COLOR_Luv2BGR", "COLOR_Luv2RGB",
    "COLOR_BGR2YCrCb", "COLOR_RGB2YCrCb", "COLOR_YCrCb2BGR", "COLOR_YCrCb2RGB",
    "COLOR_BGR2YUV", "COLOR_RGB2YUV", "COLOR_YUV2BGR", "COLOR_YUV2RGB",
    "COLOR_BGR2XYZ", "COLOR_RGB2XYZ", "COLOR_XYZ2BGR", "COLOR_XYZ2RGB",
    
    # Drawing functions
    "line", "arrowedLine", "rectangle", "circle", "ellipse",
    "polylines", "fillPoly", "fillConvexPoly", "putText", "getTextSize",
    "drawMarker", "drawContours",
    
    # Line types
    "LINE_4", "LINE_8", "LINE_AA", "FILLED",
    
    # Font constants
    "FONT_HERSHEY_SIMPLEX", "FONT_HERSHEY_PLAIN", "FONT_HERSHEY_DUPLEX",
    "FONT_HERSHEY_COMPLEX", "FONT_HERSHEY_TRIPLEX", "FONT_HERSHEY_COMPLEX_SMALL",
    "FONT_HERSHEY_SCRIPT_SIMPLEX", "FONT_HERSHEY_SCRIPT_COMPLEX", "FONT_ITALIC",
    
    # Marker types
    "MARKER_CROSS", "MARKER_TILTED_CROSS", "MARKER_STAR", "MARKER_DIAMOND",
    "MARKER_SQUARE", "MARKER_TRIANGLE_UP", "MARKER_TRIANGLE_DOWN",
    
    # Contour functions
    "findContours", "contourArea", "arcLength", "boundingRect",
    "minAreaRect", "minEnclosingCircle", "convexHull", "approxPolyDP",
    "moments", "isContourConvex", "pointPolygonTest",
    
    # Contour retrieval modes
    "RETR_EXTERNAL", "RETR_LIST", "RETR_CCOMP", "RETR_TREE", "RETR_FLOODFILL",
    
    # Contour approximation methods
    "CHAIN_APPROX_NONE", "CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_TC89_L1", "CHAIN_APPROX_TC89_KCOS",
    
    # Threshold functions
    "threshold", "adaptiveThreshold", "inRange",
    
    # Threshold types
    "THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO",
    "THRESH_TOZERO_INV", "THRESH_MASK", "THRESH_OTSU", "THRESH_TRIANGLE",
    
    # Adaptive threshold methods
    "ADAPTIVE_THRESH_MEAN_C", "ADAPTIVE_THRESH_GAUSSIAN_C",
    
    # Hough transforms (NEW)
    "HoughLines", "HoughLinesP", "HoughCircles",
    "HOUGH_STANDARD", "HOUGH_PROBABILISTIC", "HOUGH_MULTI_SCALE",
    "HOUGH_GRADIENT", "HOUGH_GRADIENT_ALT",
    
    # Geometric transforms (NEW)
    "getPerspectiveTransform", "getAffineTransform", "getRotationMatrix2D",
    "warpPerspective", "warpAffine", "remap", "invertAffineTransform",
    "perspectiveTransform", "transform",
    "INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_AREA", "INTER_LANCZOS4",
    "INTER_LINEAR_EXACT", "INTER_NEAREST_EXACT",
    "WARP_FILL_OUTLIERS", "WARP_INVERSE_MAP",
    
    # Histogram functions (NEW)
    "calcHist", "compareHist", "equalizeHist", "calcBackProject",
    "CLAHE", "createCLAHE",
    "HISTCMP_CORREL", "HISTCMP_CHISQR", "HISTCMP_INTERSECT",
    "HISTCMP_BHATTACHARYYA", "HISTCMP_HELLINGER", "HISTCMP_CHISQR_ALT", "HISTCMP_KL_DIV",
    
    # Pyramid functions (NEW)
    "pyrDown", "pyrUp", "buildPyramid",
    
    # Connected components (NEW)
    "connectedComponents", "connectedComponentsWithStats",
    "CC_STAT_LEFT", "CC_STAT_TOP", "CC_STAT_WIDTH", "CC_STAT_HEIGHT", "CC_STAT_AREA",
    
    # Template matching (NEW)
    "matchTemplate", "minMaxLoc",
    "TM_SQDIFF", "TM_SQDIFF_NORMED", "TM_CCORR", "TM_CCORR_NORMED",
    "TM_CCOEFF", "TM_CCOEFF_NORMED",
    
    # Corner detection (NEW)
    "cornerHarris", "cornerMinEigenVal", "cornerEigenValsAndVecs",
    "cornerSubPix", "preCornerDetect",
    
    # Segmentation (NEW)
    "floodFill", "distanceTransform", "distanceTransformWithLabels",
    "watershed", "grabCut", "pyrMeanShiftFiltering",
    "FLOODFILL_FIXED_RANGE", "FLOODFILL_MASK_ONLY",
    "DIST_L1", "DIST_L2", "DIST_C", "DIST_MASK_3", "DIST_MASK_5", "DIST_MASK_PRECISE",
    "GC_INIT_WITH_RECT", "GC_INIT_WITH_MASK", "GC_EVAL", "GC_EVAL_FREEZE_MODEL",
    "GC_BGD", "GC_FGD", "GC_PR_BGD", "GC_PR_FGD",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.